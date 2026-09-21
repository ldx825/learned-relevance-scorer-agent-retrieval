#!/usr/bin/env python3
"""Build format-specific NeuMF training data for Trajectory/Proceduralization.

For each format, memory units are the 23 whole-trajectory documents.  Labels
are deterministic, train-only and query-conditional: every one of the 23
trajectories spans all five phases, so the document label is derived from the
structural similarity between the query's visible signature and the document
source task's signature (analogous to the frozen V19/V25 label family).

* Grade 2: no constraint mismatch AND >= 2 matching fields (similar task shape);
* Grade 1: no mismatch with < 2 matches, or some mismatch with >= 1 match;
* Grade 0: mismatch present AND zero matching fields (conflicting task shape).

Matching fields: people_number, budget per person-day (relative tolerance 50%),
transportation / cuisine / room_type / house_rule (equal when both non-null).
Mismatching fields: people_number differs, budget out of tolerance, or the
document source task has a constraint value the query does not share.

Queries, family split and task embeddings are reused from the frozen V25 data,
so the split stays family-disjoint and no validation/test artifact is read.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_memory_embedding_candidates import embed_texts  # noqa: E402


CONSTRAINT_FIELDS = ("transportation", "cuisine", "room_type", "house_rule")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_signature(row: dict[str, str]) -> dict[str, Any]:
    constraints = ast.literal_eval(row["local_constraint"])
    days, people = int(row["days"]), int(row["people_number"])
    return {
        "days": days,
        "visiting_city_number": int(row["visiting_city_number"]),
        "people_number": people,
        "budget": int(row["budget"]),
        "budget_per_person_day": int(row["budget"]) / (people * days),
        "house_rule": constraints.get("house rule"),
        "cuisine": constraints.get("cuisine"),
        "room_type": constraints.get("room type"),
        "transportation": constraints.get("transportation"),
    }


def signature_matches(query_signature: dict[str, Any], source: dict[str, Any]) -> tuple[int, int]:
    match = mismatch = 0
    people = query_signature.get("people_number")
    if people == source["people_number"]:
        match += 1
    elif people is not None and source["people_number"] is not None:
        mismatch += 1
    budget_per_day = source["budget_per_person_day"]
    query_budget_per_day = query_signature.get("budget_per_person_day")
    if query_budget_per_day is not None:
        if abs(query_budget_per_day - budget_per_day) < 0.5 * max(budget_per_day, 1.0):
            match += 1
        else:
            mismatch += 1
    for field in CONSTRAINT_FIELDS:
        query_value = query_signature.get(field)
        source_value = source.get(field)
        if query_value and source_value and query_value == source_value:
            match += 1
        elif source_value and not query_value:
            mismatch += 1
    return match, mismatch


def grade_for(query: dict[str, Any], source: dict[str, Any]) -> int:
    match, mismatch = signature_matches(query["signature"], source)
    if mismatch == 0 and match >= 2:
        return 2
    if mismatch == 0 or match >= 1:
        return 1
    return 0


def grade_for_quality(
    query: dict[str, Any], source: dict[str, Any], quality_grade: int
) -> int:
    """V26 label: min(execution quality, task-shape match)."""
    return min(quality_grade, grade_for(query, source))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v25-data-dir", type=Path, required=True)
    parser.add_argument("--content-texts", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--format", choices=("trajectory", "proceduralization"), required=True)
    parser.add_argument(
        "--quality-bank",
        type=Path,
        help="V26 quality-gated bank jsonl; enables min(quality, match) labels",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    queries = read_jsonl(args.v25_data_dir / "phase_queries.jsonl")
    if any(row.get("validation_or_test_used") is not False for row in queries):
        raise ValueError("evaluation-derived query detected")
    if len(queries) != 1980:
        raise ValueError("expected 1980 phase queries")
    if args.quality_bank is not None:
        documents = read_jsonl(args.quality_bank)
        if len(documents) not in (41, 84):
            raise ValueError(f"expected 41 or 84 quality-bank documents, got {len(documents)}")
        quality_by_memory = {
            str(row["memory_id"]): int(row["quality_grade"]) for row in documents
        }
        if any(row.get("validation_or_test_used") is not False for row in documents):
            raise ValueError("evaluation-derived memory document detected")
    else:
        content_rows = read_jsonl(args.content_texts)
        documents = [row for row in content_rows if row["format"] == args.format]
        if len(documents) != 23:
            raise ValueError(f"expected 23 {args.format} documents")
        quality_by_memory = None
    if args.quality_bank is None:
        bank = json.loads(
            (args.bank_root / args.format / "documents.json").read_text(encoding="utf-8")
        )
        if len(bank) != 23:
            raise ValueError("bank document count mismatch")
    else:
        bank = []

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_rows = list(csv.DictReader(args.train_csv.open(encoding="utf-8", newline="")))
    signature_by_train_index = {
        int(row_index): source_signature(train_rows[int(row_index)])
        for row_index in range(len(train_rows))
    }
    item_by_train_index = {
        int(item["train_index"]): item for item in manifest["items"]
    }
    if not set(item_by_train_index).issubset(set(signature_by_train_index)):
        raise ValueError("manifest train_index out of train.csv range")

    document_meta = [
        {
            "memory_id": row["memory_id"],
            "source": row["source"],
            "format": row.get("format", args.format),
            "content_text": row["content_text"],
            "source_query": row.get("source_query"),
            "train_index": int(str(row["source"]).rsplit("_", 1)[-1]),
        }
        for row in documents
    ]
    embeddings = embed_texts(
        [str(row["content_text"])[:8000] for row in documents],
        model="text-embedding-3-small",
        cache_path=args.cache_dir / "trajectory_content_embeddings.json",
        api_base=os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1"),
        api_key=os.environ.get("SKILLDAG_EMBEDDING_API_KEY", ""),
        batch_size=64,
        timeout=180,
    ).astype(np.float32)

    with np.load(args.v25_data_dir / "arrays.npz", allow_pickle=False) as payload:
        source_arrays = {name: payload[name].copy() for name in payload.files}
    if len(source_arrays["task_embeddings"]) != len(queries):
        raise ValueError("task embedding count mismatch")
    task_index = {str(value): index for index, value in enumerate(source_arrays["subgoal_ids"])}
    if len(task_index) != len(queries):
        raise ValueError("duplicate query id")

    train_rows_out: list[dict[str, Any]] = []
    dev_rows_out: list[dict[str, Any]] = []
    pair_task: list[int] = []
    pair_memory: list[int] = []
    grades: list[int] = []
    pair_ids: list[str] = []
    for query in queries:
        query_index = task_index[str(query["subgoal_id"])]
        for memory_index, meta in enumerate(document_meta):
            source = signature_by_train_index[meta["train_index"]]
            if quality_by_memory is not None:
                grade = grade_for_quality(
                    query, source, quality_by_memory[meta["memory_id"]]
                )
            else:
                grade = grade_for(query, source)
            source_query = (
                meta["source_query"]
                if meta["source_query"]
                else item_by_train_index[meta["train_index"]]["query"]
            )
            row = {
                "subgoal_id": query["subgoal_id"],
                "family_id": query["family_id"],
                "model_split": query["model_split"],
                "subgoal_query": query["subgoal_query"],
                "phase_key": query["phase_key"],
                "memory_id": meta["memory_id"],
                "memory_format": args.format,
                "source_train_index": meta["train_index"],
                "source_query": source_query,
                "content_text": meta["content_text"][:2000],
                "final_grade": grade,
                "selection_role": f"format_{args.format}:{grade}",
                "label_source": (
                    "deterministic_train_only_quality_times_match_v1"
                    if quality_by_memory is not None
                    else "deterministic_train_only_signature_similarity_v1"
                ),
                "judge_required": False,
                "validation_or_test_used": False,
            }
            pair_task.append(query_index)
            pair_memory.append(memory_index)
            grades.append(grade)
            pair_ids.append(f"{query['subgoal_id']}::{meta['memory_id']}")
            if query["model_split"] == "internal_train":
                train_rows_out.append(row)
            else:
                dev_rows_out.append(row)

    grades_np = np.asarray(grades, dtype=np.int8)
    split_codes = np.asarray(
        [0 if row["model_split"] == "internal_train" else 1 for row in queries
         for _ in documents], dtype=np.int8)
    task_embeddings = source_arrays["task_embeddings"].astype(np.float32)
    memory_embeddings = embeddings
    task_norm = task_embeddings / np.maximum(np.linalg.norm(task_embeddings, axis=1, keepdims=True), 1e-12)
    memory_norm = memory_embeddings / np.maximum(np.linalg.norm(memory_embeddings, axis=1, keepdims=True), 1e-12)
    pair_task_np = np.asarray(pair_task, dtype=np.int32)
    pair_memory_np = np.asarray(pair_memory, dtype=np.int32)
    baseline = np.sum(task_norm[pair_task_np] * memory_norm[pair_memory_np], axis=1).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "phase_queries.jsonl", queries)
    write_jsonl(args.output_dir / "train_pairs_resolved.jsonl", train_rows_out)
    write_jsonl(args.output_dir / "internal_dev_pairs_resolved.jsonl", dev_rows_out)
    np.savez_compressed(
        args.output_dir / "arrays.npz",
        task_ids=source_arrays["task_ids"],
        memory_ids=np.asarray([row["memory_id"] for row in documents]),
        task_embeddings=task_embeddings,
        memory_embeddings=memory_embeddings,
        family_ids=source_arrays["family_ids"],
        view_types=source_arrays["view_types"],
        subgoal_ids=source_arrays["subgoal_ids"],
        operation_ids=np.asarray([row["memory_id"] for row in documents]),
        operation_parent_memory_ids=np.asarray([row["train_index"] for row in document_meta], dtype=np.int16),
        pair_ids=np.asarray(pair_ids),
        pair_task_indices=pair_task_np,
        pair_memory_indices=pair_memory_np,
        target_grade=grades_np,
        target_relevance=grades_np.astype(np.float32) / 2.0,
        label_confidence=np.ones(len(grades_np), dtype=np.float32),
        sample_weight=np.ones(len(grades_np), dtype=np.float32),
        split_codes=split_codes,
        baseline_query_cosine=baseline,
    )
    report = {
        "schema_version": f"memp.travelplanner.{args.format}_format_neumf_data.v1",
        "validation_or_test_used": False,
        "network_or_objective_changed": False,
        "sample_weighting": "uniform",
        "memory_format": args.format,
        "memory_document_count": len(documents),
        "query_count": len(queries),
        "pair_count": len(grades_np),
        "train_pair_count": len(train_rows_out),
        "dev_pair_count": len(dev_rows_out),
        "grade_counts": dict(sorted(Counter(int(value) for value in grades).items())),
        "label_rule": (
            "min(execution_quality_grade, signature_similarity_grade)"
            if quality_by_memory is not None
            else "source_task_signature_similarity_vs_query_visible_signature"
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
