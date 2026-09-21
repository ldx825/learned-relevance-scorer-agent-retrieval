#!/usr/bin/env python3
"""Embed and assemble atomic TravelPlanner data for the unchanged NeuMF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_memory_embedding_candidates import embed_texts  # noqa: E402
from prepare_travelplanner_neumf_data import load_memory_embeddings, normalized  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_embedding_records(path: Path, model: str) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model:
        raise ValueError(f"embedding cache model mismatch: {path}")
    return {
        str(record["text"]): np.asarray(record["embedding"], dtype=np.float32)
        for record in payload.get("records", {}).values()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-views", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--memory-profiles", type=Path, required=True)
    memory_group = parser.add_mutually_exclusive_group(required=True)
    memory_group.add_argument("--memory-embedding-cache", type=Path)
    memory_group.add_argument(
        "--memory-arrays",
        type=Path,
        help="Reuse the already frozen query-only Memory embeddings from an existing arrays.npz.",
    )
    parser.add_argument(
        "--query-embedding-cache", type=Path,
        help="Optional existing cache; identical query texts are reused without API calls.",
    )
    parser.add_argument(
        "--stage-query-embedding-cache",
        type=Path,
        help="Reuse frozen stage-query embeddings by exact query text.",
    )
    parser.add_argument(
        "--atomic-query-arrays",
        type=Path,
        help="Reuse frozen atomic-query embeddings indexed by source_atomic_task_id.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--train-negative-ratio", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    queries = read_jsonl(args.query_views)
    labels = read_jsonl(args.labels)
    profiles = json.loads(args.memory_profiles.read_text(encoding="utf-8"))
    if not queries or len(labels) != len(queries) * 45 or len(profiles) != 45:
        raise ValueError("expected a complete atomic-query x45-Memory relation matrix")
    if any(row.get("validation_or_test_used") is not False for row in queries + labels + profiles):
        raise ValueError("official validation/test-derived input detected")
    if any(int(row["task_id"]) != index for index, row in enumerate(queries)):
        raise ValueError("atomic query IDs must be dense and ordered")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_texts = [row["query_text"] for row in queries]
    reuse_frozen = args.stage_query_embedding_cache or args.atomic_query_arrays
    if reuse_frozen:
        if not args.stage_query_embedding_cache or not args.atomic_query_arrays:
            raise ValueError(
                "--stage-query-embedding-cache and --atomic-query-arrays must be supplied together"
            )
        stage_by_text = load_embedding_records(args.stage_query_embedding_cache, args.model)
        with np.load(args.atomic_query_arrays, allow_pickle=False) as payload:
            atomic_ids = payload["task_ids"].copy()
            atomic_embeddings = payload["task_embeddings"].copy()
        atomic_by_id = {
            int(task_id): atomic_embeddings[index]
            for index, task_id in enumerate(atomic_ids)
        }
        reused: list[np.ndarray] = []
        for query, text in zip(queries, query_texts, strict=True):
            component = query.get("candidate_matrix_component")
            if component == "stage_query_once":
                if text not in stage_by_text:
                    raise ValueError(f"stage query missing from frozen cache: task={query['task_id']}")
                reused.append(stage_by_text[text])
            elif component == "atomic_query_once":
                source_id = int(query["source_atomic_task_id"])
                if source_id not in atomic_by_id:
                    raise ValueError(f"atomic query missing from frozen arrays: source={source_id}")
                reused.append(atomic_by_id[source_id])
            else:
                raise ValueError(f"unknown compact query component: {component!r}")
        task_embeddings = np.asarray(reused, dtype=np.float32)
        api_calls_made = 0
    else:
        task_embeddings = embed_texts(
            query_texts,
            model=args.model,
            cache_path=args.query_embedding_cache or args.output_dir / "embedding_cache" / "atomic_queries.json",
            api_base=os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1"),
            api_key=os.environ.get("SKILLDAG_EMBEDDING_API_KEY", ""),
            batch_size=args.batch_size,
            timeout=args.timeout,
        )
        api_calls_made = None
    memory_texts = [row["memory_query"] for row in profiles]
    if args.memory_arrays:
        with np.load(args.memory_arrays, allow_pickle=False) as payload:
            memory_embeddings = payload["memory_embeddings"].copy()
            memory_ids_from_arrays = payload["memory_ids"].copy()
        if not np.array_equal(memory_ids_from_arrays, np.arange(len(profiles))):
            raise ValueError("frozen Memory arrays use unexpected IDs")
    else:
        memory_embeddings = load_memory_embeddings(
            args.memory_embedding_cache, memory_texts, args.model
        )
    cosine = normalized(task_embeddings) @ normalized(memory_embeddings).T

    labels_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        labels_by_task[int(row["task_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for query in queries:
        task_id = int(query["task_id"])
        rows = labels_by_task[task_id]
        determinate = [row for row in rows if row["grade"] != "uncertain"]
        positives = [row for row in determinate if row["grade"] in {1, 2}]
        negatives = sorted(
            (row for row in determinate if row["grade"] == 0),
            key=lambda row: (-float(cosine[task_id, int(row["memory_id"])]), int(row["memory_id"])),
        )
        if query["model_split"] == "internal_train":
            rows_selected = positives + negatives[: args.train_negative_ratio * len(positives)]
        else:
            rows_selected = determinate
        selected.extend(sorted(rows_selected, key=lambda row: int(row["memory_id"])))

    task_ids = np.arange(len(queries), dtype=np.int32)
    memory_ids = np.arange(len(profiles), dtype=np.int16)
    pair_count = len(selected)
    pair_task = np.empty(pair_count, dtype=np.int32)
    pair_memory = np.empty(pair_count, dtype=np.int16)
    grades = np.empty(pair_count, dtype=np.int8)
    splits = np.empty(pair_count, dtype=np.int8)
    baseline_cosine = np.empty(pair_count, dtype=np.float32)
    pair_ids: list[str] = []
    for offset, row in enumerate(selected):
        task_id = int(row["task_id"])
        memory_id = int(row["memory_id"])
        pair_task[offset] = task_id
        pair_memory[offset] = memory_id
        grades[offset] = int(row["grade"])
        splits[offset] = 0 if row["internal_split"] == "internal_train" else 1
        baseline_cosine[offset] = cosine[task_id, memory_id]
        pair_ids.append(f"atomic_{task_id:04d}::memory_{memory_id:03d}")

    np.savez_compressed(
        args.output_dir / "arrays.npz",
        task_ids=task_ids,
        memory_ids=memory_ids,
        task_embeddings=task_embeddings,
        memory_embeddings=memory_embeddings,
        family_ids=np.asarray([row["family_id"] for row in queries]),
        view_types=np.asarray([row["view_type"] for row in queries]),
        pair_ids=np.asarray(pair_ids),
        pair_task_indices=pair_task,
        pair_memory_indices=pair_memory,
        target_grade=grades,
        target_relevance=grades.astype(np.float32) / 2.0,
        label_confidence=np.ones(pair_count, dtype=np.float32),
        sample_weight=np.ones(pair_count, dtype=np.float32),
        split_codes=splits,
        baseline_query_cosine=baseline_cosine,
        candidate_source_features=np.zeros((pair_count, 5), dtype=np.float32),
    )

    # Balanced train-only anchor/rescue preferences, using the actual frozen
    # cosine ranks now that the 810 query embeddings exist.
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        if query["model_split"] != "internal_train":
            continue
        task_id = int(query["task_id"])
        rows = labels_by_task[task_id]
        order = np.argsort(-cosine[task_id], kind="stable")
        row_by_memory = {int(row["memory_id"]): row for row in rows}
        for primary in (row for row in rows if row["grade"] == 2):
            positive = int(primary["memory_id"])
            rank = int(np.flatnonzero(order == positive)[0]) + 1
            pair_type = "anchor" if rank <= 3 else "rescue"
            candidate_negatives = [
                int(memory_id) for memory_id in order
                if row_by_memory[int(memory_id)]["grade"] == 0
                and (
                    pair_type == "anchor"
                    or int(np.flatnonzero(order == memory_id)[0]) + 1 < rank
                )
            ][:3]
            for negative in candidate_negatives:
                by_type[pair_type].append(
                    {
                        "schema_version": "memp.travelplanner.atomic_preference.v1",
                        "pair_type": pair_type,
                        "task_id": task_id,
                        "positive_memory_id": positive,
                        "negative_memory_id": negative,
                        "source_split": "internal_train",
                        "validation_or_test_used": False,
                    }
                )
    budget = min(len(by_type["anchor"]), len(by_type["rescue"]))
    balanced = sorted(by_type["anchor"], key=lambda row: (row["task_id"], row["negative_memory_id"]))[:budget]
    balanced += sorted(by_type["rescue"], key=lambda row: (row["task_id"], row["negative_memory_id"]))[:budget]
    write_jsonl(args.output_dir / "balanced_preferences.jsonl", balanced)

    report = {
        "schema_version": "memp.travelplanner.atomic_neumf_data.v1",
        "api_query_embedding_count": len(queries) if api_calls_made is None else 0,
        "api_calls_made": api_calls_made,
        "memory_embedding_recomputed": False,
        "embedding_dimension": int(task_embeddings.shape[1]),
        "query_count": len(queries),
        "memory_count": len(profiles),
        "selected_pair_count": pair_count,
        "selected_pair_counts_by_split": {
            "internal_train": int(np.sum(splits == 0)),
            "internal_dev": int(np.sum(splits == 1)),
        },
        "selected_grade_counts": dict(Counter(str(int(value)) for value in grades)),
        "preference_counts_raw": {key: len(value) for key, value in by_type.items()},
        "balanced_preference_count": len(balanced),
        "model_inputs": ["atomic_task_query_embedding", "memory_source_query_embedding"],
        "model_or_fusion_changed": False,
        "validation_or_test_used": False,
        "integrity": {
            "query_views": sha256(args.query_views),
            "labels": sha256(args.labels),
            "memory_profiles": sha256(args.memory_profiles),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
