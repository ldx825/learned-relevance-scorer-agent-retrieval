#!/usr/bin/env python3
"""Build query-only, multi-view Memory-Task candidates from train data."""

from __future__ import annotations

import argparse
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

from build_memory_embedding_candidates import embed_texts, normalized  # noqa: E402
from memory_task_schema import TaskSignature, negative_hardness, relation_grade  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def signature(payload: dict[str, Any]) -> TaskSignature:
    return TaskSignature(
        operation=str(payload["operation"]),
        object_type=str(payload["object_type"]),
        cardinality=int(payload["cardinality"]),
        destination=str(payload["destination"]),
    )


def view_relation(
    view_type: str,
    task: TaskSignature,
    memory: TaskSignature,
) -> tuple[int, str]:
    """Conservative seed label for a focused query view.

    These labels only control candidate coverage and deterministic calibration;
    the grouped LLM Judge remains the source of the final 2/1/0 label.
    """
    if view_type == "full_task":
        return relation_grade(task, memory, memory_is_valid=True)

    same_operation = task.operation == memory.operation
    same_object = task.object_type == memory.object_type
    same_cardinality = task.cardinality == memory.cardinality
    same_destination = task.destination == memory.destination

    if view_type == "operation":
        if same_operation and same_cardinality:
            return 2, "focused_operation_and_cardinality_match"
        if (
            task.operation in {"clean", "heat", "cool", "pick_two"}
            and memory.operation == "pick_place"
            and same_cardinality
            and (same_object or same_destination)
        ):
            return 1, "focused_placement_subprocedure"
        if same_object or same_destination or task.operation != memory.operation:
            return 0, "focused_operation_conflict"
        return 0, "focused_operation_unrelated"

    if view_type == "object":
        if same_object and same_cardinality:
            return 2, "focused_object_and_cardinality_match"
        if same_object:
            return 1, "focused_object_partial_cardinality"
        if same_operation or same_destination:
            return 0, "focused_object_hard_negative"
        return 0, "focused_object_unrelated"

    if view_type == "destination":
        if same_destination and same_cardinality:
            return 2, "focused_destination_and_cardinality_match"
        if same_destination:
            return 1, "focused_destination_partial_cardinality"
        if same_operation or same_object:
            return 0, "focused_destination_hard_negative"
        return 0, "focused_destination_unrelated"

    raise ValueError(f"unsupported view_type: {view_type}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--query-views", type=Path, required=True)
    parser.add_argument("--baseline-memory-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=24)
    parser.add_argument("--cosine-top", type=int, default=12)
    parser.add_argument("--per-role", type=int, default=2)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--reuse-parent-query-embedding",
        action="store_true",
        help=(
            "Reuse the paper Query-cosine embedding of the unmodified parent task for every "
            "focused view. This is an offline fallback when focused-query embeddings are unavailable."
        ),
    )
    args = parser.parse_args()

    views = load_jsonl(args.query_views)
    memories = load_jsonl(args.dataset / "memories.jsonl")
    api_key = os.environ.get("SKILLDAG_EMBEDDING_API_KEY", "")
    api_base = os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1")
    embedding_texts = [
        row["task_query"] if args.reuse_parent_query_embedding else row["query_text"]
        for row in views
    ]
    view_embeddings = embed_texts(
        embedding_texts,
        model=args.embedding_model,
        cache_path=args.output_dir / "embedding_cache" / "query_views.json",
        api_base=api_base,
        api_key=api_key,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    baseline = json.loads(args.baseline_memory_embeddings.read_text(encoding="utf-8"))
    if baseline.get("model") != args.embedding_model:
        raise ValueError("memory embedding model mismatch")
    expected_queries = [row["query"] for row in memories]
    if baseline.get("queries") != expected_queries:
        raise ValueError("memory query ordering differs from the frozen bank")
    memory_embeddings = np.asarray(baseline["embeddings"], dtype=np.float32)
    scores = normalized(view_embeddings) @ normalized(memory_embeddings).T

    memory_signatures = [signature(row["signature"]) for row in memories]
    batches: list[dict[str, Any]] = []
    grade_counts: Counter[int] = Counter()
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for view_offset, view in enumerate(views):
        task_signature = signature(view["signature"])
        relation_rows: list[dict[str, Any]] = []
        for memory_id, memory_signature in enumerate(memory_signatures):
            grade, reason = view_relation(view["view_type"], task_signature, memory_signature)
            relation_rows.append(
                {
                    "memory_id": memory_id,
                    "rule_seed_grade": grade,
                    "rule_seed_reason": reason,
                    "hardness": negative_hardness(task_signature, memory_signature),
                }
            )
        relation_by_id = {row["memory_id"]: row for row in relation_rows}
        cosine_order = np.argsort(-scores[view_offset], kind="stable")
        cosine_rank = {int(memory_id): rank for rank, memory_id in enumerate(cosine_order, 1)}

        by_grade_role: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in relation_rows:
            by_grade_role[int(row["rule_seed_grade"])][row["rule_seed_reason"]].append(row)
        for roles in by_grade_role.values():
            for rows in roles.values():
                rows.sort(key=lambda row: (cosine_rank[row["memory_id"]], -row["hardness"], row["memory_id"]))

        selected: list[int] = []
        selected_sources: dict[int, set[str]] = defaultdict(set)

        def add(memory_id: int, source: str) -> None:
            selected_sources[memory_id].add(source)
            if memory_id not in selected and len(selected) < args.candidate_budget:
                selected.append(memory_id)

        # Guarantee direct, complementary and confusing examples before
        # adding the live Query-cosine route that the NCF must rerank.
        for grade, source in ((2, "structured_primary"), (1, "structured_support")):
            for role in sorted(by_grade_role[grade]):
                for row in by_grade_role[grade][role][: args.per_role]:
                    add(int(row["memory_id"]), source)
        for role in sorted(by_grade_role[0]):
            for row in by_grade_role[0][role][: args.per_role]:
                add(int(row["memory_id"]), "structured_hard_negative")
        for memory_id in cosine_order[: args.cosine_top]:
            add(int(memory_id), "query_cosine")
        for memory_id in cosine_order:
            if len(selected) >= args.candidate_budget:
                break
            add(int(memory_id), "query_cosine_fill")

        candidates = []
        for memory_id in sorted(selected):
            relation = relation_by_id[memory_id]
            sources = sorted(selected_sources[memory_id])
            candidates.append(
                {
                    "memory_id": memory_id,
                    "memory_query": memories[memory_id]["query"],
                    "candidate_sources": sources,
                    "query_cosine_rank": cosine_rank[memory_id],
                    "query_cosine_score": float(scores[view_offset, memory_id]),
                    "rule_seed_grade": relation["rule_seed_grade"],
                    "rule_seed_reason": relation["rule_seed_reason"],
                    "relation_template_id": "|".join(
                        (
                            str(view["view_type"]),
                            task_signature.operation,
                            str(task_signature.cardinality),
                            str(relation["rule_seed_reason"]),
                        )
                    ),
                }
            )
            grade_counts[int(relation["rule_seed_grade"])] += 1
            reason_counts[str(relation["rule_seed_reason"])] += 1
            source_counts.update(sources)
        if len(candidates) != args.candidate_budget:
            raise ValueError(f"candidate budget mismatch: {view['query_view_id']}")
        batches.append(
            {
                "schema_version": "task_resource_ncf.memory_query_only_candidate.v1",
                "task_id": int(view["query_view_index"]),
                "query_view_id": view["query_view_id"],
                "parent_task_id": int(view["parent_task_id"]),
                "view_type": view["view_type"],
                "task_query": view["query_text"],
                "task_signature": view["signature"],
                "canonical_id": view["canonical_id"],
                "internal_split": view["model_split"],
                "candidates": candidates,
                "judge_evidence_fields": ["task_query", "memory_query"],
                "uses_workflow_text": False,
                "uses_eval_data": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "judge_candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in batches),
        encoding="utf-8",
    )
    report = {
        "schema_version": "task_resource_ncf.memory_query_only_candidates_report.v1",
        "query_view_count": len(views),
        "memory_count": len(memories),
        "candidate_budget": args.candidate_budget,
        "candidate_pair_count": len(views) * args.candidate_budget,
        "rule_seed_grade_counts": {str(key): grade_counts[key] for key in (2, 1, 0)},
        "rule_seed_reason_counts": dict(sorted(reason_counts.items())),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "embedding_model": args.embedding_model,
        "query_embedding_text": (
            "unmodified_parent_task_query"
            if args.reuse_parent_query_embedding
            else "focused_query_view"
        ),
        "model_evidence": ["task_query", "memory_source_query"],
        "workflow_text_used": False,
        "official_dev_or_test_read": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
