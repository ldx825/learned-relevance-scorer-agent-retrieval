#!/usr/bin/env python3
"""Embed and assemble task-subgoal--atomic-operation NeuMF arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_memory_embedding_candidates import embed_texts  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgoals", type=Path, required=True)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--dev-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subgoals = read_jsonl(args.subgoals)
    operations = read_jsonl(args.operations)
    train_pairs = read_jsonl(args.train_pairs)
    dev_pairs = read_jsonl(args.dev_pairs)
    pairs = train_pairs + dev_pairs
    if not subgoals or not operations or not pairs:
        raise ValueError("empty atomic-operation input")
    if any(row.get("validation_or_test_used") is not False for row in subgoals + operations + pairs):
        raise ValueError("official validation/test-derived input detected")
    if any(row.get("model_split") != "internal_train" for row in train_pairs):
        raise ValueError("non-train pair in train file")
    if any(row.get("model_split") != "internal_dev" for row in dev_pairs):
        raise ValueError("non-dev pair in dev file")
    if any(row.get("final_grade") == "uncertain" for row in pairs):
        raise ValueError("unresolved label in train/dev pair files")

    subgoal_index = {str(row["subgoal_id"]): index for index, row in enumerate(subgoals)}
    operation_index = {str(row["operation_id"]): index for index, row in enumerate(operations)}
    if len(subgoal_index) != len(subgoals) or len(operation_index) != len(operations):
        raise ValueError("duplicate subgoal or operation ID")
    pair_keys = [(str(row["subgoal_id"]), str(row["operation_id"])) for row in pairs]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("duplicate task-subgoal--operation pair")
    if any(subgoal_id not in subgoal_index or operation_id not in operation_index for subgoal_id, operation_id in pair_keys):
        raise ValueError("pair references unknown subgoal or operation")

    train_families = {str(row["family_id"]) for row in train_pairs}
    dev_families = {str(row["family_id"]) for row in dev_pairs}
    if train_families & dev_families:
        raise ValueError("family leakage across internal train/dev")

    report_base = {
        "schema_version": "memp.travelplanner.atomic_operation_neumf_data.v1",
        "embedding_model": args.model,
        "subgoal_count": len(subgoals),
        "operation_count": len(operations),
        "train_pair_count": len(train_pairs),
        "dev_pair_count": len(dev_pairs),
        "grade_counts_by_split": {
            "internal_train": dict(Counter(str(row["final_grade"]) for row in train_pairs)),
            "internal_dev": dict(Counter(str(row["final_grade"]) for row in dev_pairs)),
        },
        "train_family_count": len(train_families),
        "dev_family_count": len(dev_families),
        "cross_split_family_count": 0,
        "validation_or_test_used": False,
    }
    if args.dry_run:
        print(json.dumps({**report_base, "embedding_text_count": len(subgoals) + len(operations)}, ensure_ascii=False, indent=2))
        return 0

    api_base = os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1")
    api_key = os.environ.get("SKILLDAG_EMBEDDING_API_KEY", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "embedding_cache"
    task_embeddings = embed_texts(
        [str(row["subgoal_query"]) for row in subgoals],
        model=args.model,
        cache_path=cache / "subgoal_queries.json",
        api_base=api_base,
        api_key=api_key,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    operation_embeddings = embed_texts(
        [str(row["operation_text"]) for row in operations],
        model=args.model,
        cache_path=cache / "operations.json",
        api_base=api_base,
        api_key=api_key,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )

    pair_task = np.asarray([subgoal_index[str(row["subgoal_id"])] for row in pairs], dtype=np.int32)
    pair_operation = np.asarray([operation_index[str(row["operation_id"])] for row in pairs], dtype=np.int32)
    grades = np.asarray([int(row["final_grade"]) for row in pairs], dtype=np.int8)
    split_codes = np.asarray([0] * len(train_pairs) + [1] * len(dev_pairs), dtype=np.int8)
    normalized_tasks = task_embeddings / np.maximum(np.linalg.norm(task_embeddings, axis=1, keepdims=True), 1e-12)
    normalized_operations = operation_embeddings / np.maximum(np.linalg.norm(operation_embeddings, axis=1, keepdims=True), 1e-12)
    baseline_cosine = np.sum(normalized_tasks[pair_task] * normalized_operations[pair_operation], axis=1).astype(np.float32)

    np.savez_compressed(
        args.output_dir / "arrays.npz",
        task_ids=np.arange(len(subgoals), dtype=np.int32),
        memory_ids=np.arange(len(operations), dtype=np.int32),
        task_embeddings=task_embeddings.astype(np.float32),
        memory_embeddings=operation_embeddings.astype(np.float32),
        family_ids=np.asarray([str(row["family_id"]) for row in subgoals]),
        view_types=np.asarray([str(row["operation_key"]) for row in subgoals]),
        subgoal_ids=np.asarray([str(row["subgoal_id"]) for row in subgoals]),
        operation_ids=np.asarray([str(row["operation_id"]) for row in operations]),
        operation_parent_memory_ids=np.asarray([int(row["parent_memory_id"]) for row in operations], dtype=np.int16),
        pair_ids=np.asarray([f"{left}::{right}" for left, right in pair_keys]),
        pair_task_indices=pair_task,
        pair_memory_indices=pair_operation,
        target_grade=grades,
        target_relevance=grades.astype(np.float32) / 2.0,
        label_confidence=np.ones(len(pairs), dtype=np.float32),
        sample_weight=np.ones(len(pairs), dtype=np.float32),
        split_codes=split_codes,
        baseline_query_cosine=baseline_cosine,
    )

    report = {
        **report_base,
        "embedding_dimension": int(task_embeddings.shape[1]),
        "embedding_text_count": len(subgoals) + len(operations),
        "sample_weighting": "uniform",
        "target_relevance": "grade / 2",
        "network_or_objective_changed": False,
        "integrity": {
            "subgoals": digest(args.subgoals),
            "operations": digest(args.operations),
            "train_pairs": digest(args.train_pairs),
            "dev_pairs": digest(args.dev_pairs),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
