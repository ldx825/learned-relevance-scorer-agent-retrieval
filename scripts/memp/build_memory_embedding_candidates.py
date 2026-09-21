#!/usr/bin/env python3
"""Embed train-only MemP texts and build unified multi-route Judge candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src/SkillDAG_NCF"
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from skilldag.initialize import _http_post_json  # noqa: E402
from memory_task_schema import (  # noqa: E402
    negative_hardness,
    parse_query,
    relation_grade,
    script_validity,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def text_key(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_texts(
    texts: list[str],
    *,
    model: str,
    cache_path: Path,
    api_base: str,
    api_key: str,
    batch_size: int,
    timeout: int,
) -> np.ndarray:
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("model") != model:
            raise ValueError(f"embedding cache model mismatch: {cache_path}")
    else:
        cache = {"model": model, "records": {}}
    records = cache["records"]
    missing = [(text_key(text), text) for text in texts if text_key(text) not in records]
    if missing and not api_key:
        raise RuntimeError(
            f"{len(missing)} embedding(s) are absent from {cache_path}; "
            "set SKILLDAG_EMBEDDING_API_KEY before making a paid API call"
        )
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                status, body = _http_post_json(
                    api_base.rstrip("/") + "/embeddings",
                    {"Authorization": f"Bearer {api_key}"},
                    {"model": model, "input": [text for _, text in batch]},
                    timeout,
                )
                if status >= 400:
                    raise RuntimeError(f"Embedding HTTP {status}: {body[:500]}")
                break
            except Exception as exc:
                last_error = exc
                if attempt == 5:
                    raise
                delay = min(2 ** (attempt - 1), 8)
                print(
                    f"[embed] transient error attempt={attempt}/5; retrying in {delay}s: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(delay)
        else:  # pragma: no cover - defensive; the loop either breaks or raises.
            raise RuntimeError(f"embedding request failed: {last_error}")
        response = json.loads(body)
        values = sorted(response.get("data") or [], key=lambda row: int(row["index"]))
        if len(values) != len(batch):
            raise ValueError(f"embedding count mismatch: expected {len(batch)}, got {len(values)}")
        for (key, text), value in zip(batch, values):
            records[key] = {"text": text, "embedding": value["embedding"]}
        atomic_json(cache_path, cache)
        print(f"[embed] model={model} completed={min(offset + len(batch), len(missing))}/{len(missing)}", flush=True)
    return np.asarray([records[text_key(text)]["embedding"] for text in texts], dtype=np.float32)


def normalized(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def add_source(pool: dict[int, dict], memory_id: int, source: str, **metadata: Any) -> None:
    row = pool.setdefault(memory_id, {"memory_id": memory_id, "candidate_sources": []})
    if source not in row["candidate_sources"]:
        row["candidate_sources"].append(source)
    row.update(metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-memory-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=24)
    parser.add_argument("--baseline-top", type=int, default=8)
    parser.add_argument("--content-top", type=int, default=8)
    parser.add_argument("--per-relation-role", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--small-model", default="text-embedding-3-small")
    parser.add_argument("--large-model", default="text-embedding-3-large")
    args = parser.parse_args()

    api_key = os.environ.get("SKILLDAG_EMBEDDING_API_KEY", "")
    api_base = os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1")
    # A key is only needed when a cache entry is missing.  Fully cached
    # candidate rebuilding is deliberately offline and incurs no API cost.

    tasks = load_jsonl(args.dataset / "tasks.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    task_texts = [row["query"] for row in tasks]
    memory_content_texts = [
        f"Memory task: {row['query']}\nReusable procedure: {row['workflow']}"
        for row in memories
    ]

    cache_dir = args.output_dir / "embedding_cache"
    task_small = embed_texts(
        task_texts,
        model=args.small_model,
        cache_path=cache_dir / "task_query_small.json",
        api_base=api_base,
        api_key=api_key,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    task_large = embed_texts(
        task_texts,
        model=args.large_model,
        cache_path=cache_dir / "task_query_large.json",
        api_base=api_base,
        api_key=api_key,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    memory_large = embed_texts(
        memory_content_texts,
        model=args.large_model,
        cache_path=cache_dir / "memory_content_large.json",
        api_base=api_base,
        api_key=api_key,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )

    baseline_payload = json.loads(args.baseline_memory_embeddings.read_text(encoding="utf-8"))
    if baseline_payload.get("model") != args.small_model:
        raise ValueError("baseline memory embedding model does not match small model")
    baseline_queries = [row["query"] for row in memories]
    if baseline_payload.get("queries") != baseline_queries:
        raise ValueError("baseline memory embedding order/text differs from frozen memory bank")
    memory_small = np.asarray(baseline_payload["embeddings"], dtype=np.float32)

    baseline_scores = normalized(task_small) @ normalized(memory_small).T
    content_scores = normalized(task_large) @ normalized(memory_large).T
    batches = []
    source_counts: dict[str, int] = {}
    for task_offset, task in enumerate(tasks):
        signature = parse_query(task["query"])
        relations = []
        for memory in memories:
            memory_signature = parse_query(memory["query"])
            valid, validity_reason = script_validity(memory_signature, memory["workflow"])
            grade, reason = relation_grade(signature, memory_signature, memory_is_valid=valid)
            relations.append({
                "memory_id": int(memory["memory_id"]),
                "rule_seed_grade": grade,
                "rule_seed_reason": reason,
                "procedurally_valid_rule": valid,
                "validity_reason_rule": validity_reason,
                "hardness": negative_hardness(signature, memory_signature),
            })

        pool: dict[int, dict] = {}
        baseline_order = np.argsort(-baseline_scores[task_offset], kind="stable")
        content_order = np.argsort(-content_scores[task_offset], kind="stable")
        for rank, memory_id in enumerate(baseline_order[: args.baseline_top], 1):
            add_source(pool, int(memory_id), "query_cosine", query_cosine_rank=rank, query_cosine_score=float(baseline_scores[task_offset, memory_id]))
        for rank, memory_id in enumerate(content_order[: args.content_top], 1):
            add_source(pool, int(memory_id), "resource_content_cosine", content_cosine_rank=rank, content_cosine_score=float(content_scores[task_offset, memory_id]))

        exact = [row for row in relations if row["rule_seed_grade"] == 2]
        complement = [row for row in relations if row["rule_seed_grade"] == 1]
        negatives = [row for row in relations if row["rule_seed_grade"] == 0]

        def relation_key(row: dict[str, Any]) -> tuple[float, float, int]:
            memory_id = row["memory_id"]
            return (
                -max(
                    float(baseline_scores[task_offset, memory_id]),
                    float(content_scores[task_offset, memory_id]),
                ),
                -float(row["hardness"]),
                int(memory_id),
            )

        exact.sort(key=relation_key)
        complement_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        negative_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in complement:
            complement_by_role[row["rule_seed_reason"]].append(row)
        for row in negatives:
            role = (
                "procedurally_invalid"
                if not row["procedurally_valid_rule"]
                else row["rule_seed_reason"]
            )
            negative_by_role[role].append(row)
        for rows in (*complement_by_role.values(), *negative_by_role.values()):
            rows.sort(key=relation_key)

        # Failure-targeted, role-balanced selection.  It guarantees that the
        # grouped Judge sees exact memories, distinct complementary procedures,
        # and confusing operation/cardinality/validity negatives for each train
        # task.  No official dev/test query or trajectory is consulted.
        selected_ids: list[int] = []

        def select(row: dict[str, Any], source: str) -> None:
            memory_id = int(row["memory_id"])
            add_source(pool, memory_id, source)
            if memory_id not in selected_ids and len(selected_ids) < args.candidate_budget:
                selected_ids.append(memory_id)

        for row in exact[:4]:
            select(row, "structured_exact")
        # First retain one representative of every distinct positive role;
        # only then add a second representative for robust Judge comparison.
        for role_offset in range(args.per_relation_role):
            for role in sorted(complement_by_role):
                rows = complement_by_role[role]
                if role_offset < len(rows):
                    select(rows[role_offset], "structured_complement")
        # The first hard negative from every relation-conflict class prevents
        # the model from learning only easy random negatives.
        for role in sorted(negative_by_role):
            if negative_by_role[role]:
                select(negative_by_role[role][0], "structured_hard_negative")

        # Preserve the paper Query-cosine route and the content route.  These
        # are the candidates whose ordering the plug-in must actually improve.
        for rank in range(max(args.baseline_top, args.content_top)):
            if rank < args.baseline_top:
                memory_id = int(baseline_order[rank])
                if memory_id not in selected_ids and len(selected_ids) < args.candidate_budget:
                    selected_ids.append(memory_id)
            if rank < args.content_top:
                memory_id = int(content_order[rank])
                if memory_id not in selected_ids and len(selected_ids) < args.candidate_budget:
                    selected_ids.append(memory_id)

        # Add a second hard example per role, then fill deterministically with
        # the globally most confusable remaining negatives.
        for role_offset in range(1, args.per_relation_role):
            for role in sorted(negative_by_role):
                rows = negative_by_role[role]
                if role_offset < len(rows):
                    select(rows[role_offset], "structured_hard_negative")
        for row in sorted(negatives, key=relation_key):
            select(row, "structured_hard_negative")
            if len(selected_ids) >= args.candidate_budget:
                break
        relation_by_id = {row["memory_id"]: row for row in relations}
        candidate_rows = []
        for memory_id in sorted(selected_ids):
            memory = memories[memory_id]
            relation = relation_by_id[memory_id]
            candidate = {
                **pool[memory_id],
                "memory_query": memory["query"],
                "workflow": memory["workflow"],
                **{key: relation[key] for key in ("rule_seed_grade", "rule_seed_reason", "procedurally_valid_rule", "validity_reason_rule")},
            }
            candidate_rows.append(candidate)
            for source in candidate["candidate_sources"]:
                source_counts[source] = source_counts.get(source, 0) + 1
        if len(candidate_rows) != args.candidate_budget:
            raise ValueError(f"candidate budget mismatch for task {task['task_id']}")
        batches.append({
            "schema_version": "task_resource_ncf.memory_judge_candidate.v3",
            "task_id": task["task_id"],
            "task_query": task["query"],
            "task_signature": task["signature"],
            "canonical_id": task["canonical_id"],
            "internal_split": task["model_split"],
            "source": task["source"],
            "candidates": candidate_rows,
            "contains_final_labels": False,
            "uses_eval_data": False,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "judge_candidates.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in batches), encoding="utf-8")
    report = {
        "schema_version": "task_resource_ncf.memory_embedding_candidates_report.v2",
        "task_count": len(tasks),
        "memory_count": len(memories),
        "candidate_pair_count": len(tasks) * args.candidate_budget,
        "candidate_budget": args.candidate_budget,
        "selection_policy": {
            "name": "train_only_failure_mode_role_balanced",
            "per_relation_role": args.per_relation_role,
            "derived_from_eval_examples": False,
            "generic_failure_modes": [
                "operation_or_cardinality_conflict",
                "complementary_object_destination_procedures",
                "reusable_operation_template",
                "procedurally_invalid_memory",
            ],
        },
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "embedding_models": {"baseline_query": args.small_model, "content_neumf": args.large_model},
        "embedding_text_counts": {"task_small": len(tasks), "task_large": len(tasks), "memory_content_large": len(memories)},
        "uses_eval_data": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
