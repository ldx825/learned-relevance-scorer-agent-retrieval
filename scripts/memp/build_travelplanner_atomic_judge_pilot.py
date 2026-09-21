#!/usr/bin/env python3
"""Build a stratified operation-level Judge pilot from train-only candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--mode", choices=("pilot", "all-uncertain-train"), default="pilot")
    args = parser.parse_args()
    operations = {row["operation_id"]: row for row in read_jsonl(args.data_dir / "memory_operations.jsonl")}
    subgoals = {row["subgoal_id"]: row for row in read_jsonl(args.data_dir / "task_subgoals.jsonl")}
    pairs_by_subgoal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.data_dir / "candidate_pairs.jsonl"):
        if row["model_split"] == "internal_train":
            pairs_by_subgoal[row["subgoal_id"]].append(row)

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for subgoal_id, rows in pairs_by_subgoal.items():
        key = subgoals[subgoal_id]["operation_key"]
        if not any(row["seed_grade"] == 2 for row in rows):
            kind = "missing_grade2"
        elif any(row["selection_role"] == "explicit_conflict" for row in rows):
            kind = "explicit_conflict"
        else:
            kind = "regular"
        buckets[(kind, key)].append(subgoal_id)

    if args.mode == "all-uncertain-train":
        selected = sorted(
            subgoal_id for subgoal_id, rows in pairs_by_subgoal.items()
            if any(row["seed_grade"] == "uncertain" for row in rows)
        )
    else:
        selected = []
    # Hard transfer/conflict groups come first; then fill round-robin across
    # operation keys so the pilot cannot be dominated by generic stages.
    for kind in (() if args.mode == "all-uncertain-train" else ("missing_grade2", "explicit_conflict", "regular")):
        keys = sorted(key for bucket_kind, key in buckets if bucket_kind == kind)
        cursor = 0
        while keys and len(selected) < args.count:
            key = keys[cursor % len(keys)]
            rows = buckets[(kind, key)]
            if rows:
                candidate = rows.pop(0)
                if candidate not in selected:
                    selected.append(candidate)
            keys = [value for value in keys if buckets[(kind, value)]]
            cursor += 1
            if kind == "missing_grade2" and len(selected) >= min(10, args.count):
                break
        if len(selected) >= args.count:
            break

    batches = []
    for task_id, subgoal_id in enumerate(selected):
        subgoal = subgoals[subgoal_id]
        rows = pairs_by_subgoal[subgoal_id]
        batches.append(
            {
                "task_id": task_id,
                "subgoal_id": subgoal_id,
                "family_id": subgoal["family_id"],
                "internal_split": subgoal["model_split"],
                "full_task_query": subgoal["task_query"],
                "subgoal_query": subgoal["subgoal_query"],
                "required_operation_key": subgoal["operation_key"],
                "required_value": subgoal.get("required_value"),
                "candidates": [
                    {
                        "operation_id": row["operation_id"],
                        "parent_memory_id": row["parent_memory_id"],
                        "parent_memory_query": operations[row["operation_id"]]["parent_memory_query"],
                        "operation_key": operations[row["operation_id"]]["operation_key"],
                        "operation_family": operations[row["operation_id"]]["operation_family"],
                        "applicability_value": operations[row["operation_id"]].get("applicability_value"),
                        "operation_text": operations[row["operation_id"]]["operation_text"],
                    }
                    for row in rows
                ],
                "seed_fields_visible_to_judge": False,
                "validation_or_test_used": False,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in batches),
        encoding="utf-8",
    )
    print(json.dumps({
        "pilot_group_count": len(batches),
        "pilot_pair_count": sum(len(row["candidates"]) for row in batches),
        "operation_key_counts": {
            key: sum(row["required_operation_key"] == key for row in batches)
            for key in sorted({row["required_operation_key"] for row in batches})
        },
        "validation_or_test_used": False,
        "mode": args.mode,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
