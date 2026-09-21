#!/usr/bin/env python3
"""Build a no-API pilot for operation-level TravelPlanner Memory retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from travelplanner_atomic_memory import decompose_memory, decompose_task, seed_grade


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def structural_distance(task: dict[str, Any], memory: dict[str, Any]) -> float:
    return (
        abs(int(task["days"]) - int(memory["days"])) / 7.0
        + abs(int(task["visiting_city_number"]) - int(memory["visiting_city_number"])) / 3.0
        + abs(int(task["people_number"]) - int(memory["people_number"])) / 10.0
        + abs(float(task["budget_per_person_day"]) - float(memory["budget_per_person_day"]))
        / max(float(task["budget_per_person_day"]), 1.0)
    )


def select_informative_operations(
    subgoal: dict[str, Any], operations: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], int | str, str, str]]:
    """Keep one direct anchor, complements, and one hard negative per subgoal."""
    buckets: dict[str, list[tuple[dict[str, Any], int | str, str]]] = defaultdict(list)
    for operation in operations:
        grade, reason = seed_grade(subgoal, operation)
        if grade == 2:
            bucket = "direct_anchor"
        elif operation["operation_key"] == subgoal["operation_key"]:
            bucket = "explicit_conflict"
        elif operation["operation_family"] == subgoal["operation_family"]:
            bucket = "same_family_complement"
        elif grade == 1:
            bucket = "verification_complement"
        else:
            bucket = "different_family_negative"
        buckets[bucket].append((operation, grade, reason))

    def ranked(bucket: str) -> list[tuple[dict[str, Any], int | str, str]]:
        return sorted(
            buckets[bucket],
            key=lambda item: (
                structural_distance(subgoal["signature"], item[0]["source_signature"]),
                item[0]["parent_memory_id"], item[0]["order"],
            ),
        )

    chosen: list[tuple[dict[str, Any], int | str, str, str]] = []
    # Exactly one strongest direct operation is the groupwise Grade-2 anchor.
    if ranked("direct_anchor"):
        operation, grade, reason = ranked("direct_anchor")[0]
        chosen.append((operation, grade, reason, "direct_anchor"))
    # Complementarity is learned without flooding the group with generic G1s.
    for bucket in ("same_family_complement", "verification_complement"):
        if ranked(bucket):
            operation, grade, reason = ranked(bucket)[0]
            chosen.append((operation, grade, reason, bucket))
    # Prefer a deceptively related explicit conflict; otherwise retain one
    # structurally close different-family negative.
    negative_bucket = "explicit_conflict" if ranked("explicit_conflict") else "different_family_negative"
    if ranked(negative_bucket):
        operation, grade, reason = ranked(negative_bucket)[0]
        chosen.append((operation, grade, reason, negative_bucket))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-profiles", type=Path, required=True)
    parser.add_argument("--task-queries", type=Path, nargs="+", required=True)
    parser.add_argument("--memory-operations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", type=int, default=5)
    parser.add_argument("--task-limit", type=int, default=20)
    args = parser.parse_args()

    payload = json.loads(args.memory_profiles.read_text(encoding="utf-8"))
    profiles = payload["profiles"] if isinstance(payload, dict) else payload
    tasks = []
    seen_queries: set[str] = set()
    for path in args.task_queries:
        for row in read_jsonl(path):
            if row.get("view_type") not in {None, "full_task"}:
                continue
            query = (row.get("task_query") or row.get("query_text") or "").strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            tasks.append(row)
    if any(row.get("validation_or_test_used") is not False for row in profiles + tasks):
        raise ValueError("validation/test-derived input detected")
    profiles = profiles[: args.memory_limit]
    tasks = tasks[: args.task_limit]

    if args.memory_operations:
        operations = [
            row for row in read_jsonl(args.memory_operations)
            if int(row["parent_memory_id"]) in {int(profile["memory_id"]) for profile in profiles}
        ]
    else:
        operations = [operation for profile in profiles for operation in decompose_memory(profile)]
    subgoals: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        family_id = task.get("family_id", f"travelplanner_atomic_pilot_{index:04d}")
        split = (
            "internal_dev"
            if int(hashlib.sha256(family_id.encode()).hexdigest(), 16) % 5 == 0
            else "internal_train"
        )
        subgoals.extend(
            decompose_task(
                task.get("task_query") or task["query_text"],
                task["signature"],
                family_id=family_id,
                model_split=split,
            )
        )

    pairs: list[dict[str, Any]] = []
    for task_id, subgoal in enumerate(subgoals):
        for operation, grade, reason, selection_role in select_informative_operations(subgoal, operations):
            pairs.append(
                {
                    "task_id": task_id,
                    "subgoal_id": subgoal["subgoal_id"],
                    "family_id": subgoal["family_id"],
                    "model_split": subgoal["model_split"],
                    "subgoal_query": subgoal["subgoal_query"],
                    "operation_id": operation["operation_id"],
                    "parent_memory_id": operation["parent_memory_id"],
                    "operation_text": operation["operation_text"],
                    "seed_grade": grade,
                    "seed_reason": reason,
                    "selection_role": selection_role,
                    "judge_required": grade in {1, 2, "uncertain"},
                    "validation_or_test_used": False,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "memory_operations.jsonl", operations)
    write_jsonl(args.output_dir / "task_subgoals.jsonl", subgoals)
    write_jsonl(args.output_dir / "candidate_pairs.jsonl", pairs)
    report = {
        "schema_version": "memp.travelplanner.atomic_memory_pilot.v1",
        "parent_memory_count": len(profiles),
        "task_count": len(tasks),
        "operation_count": len(operations),
        "subgoal_count": len(subgoals),
        "candidate_pair_count": len(pairs),
        "operation_counts_by_key": dict(Counter(row["operation_key"] for row in operations)),
        "subgoal_counts_by_key": dict(Counter(row["operation_key"] for row in subgoals)),
        "task_family_counts_by_split": dict(Counter(
            row["model_split"] for row in subgoals
            if row["operation_key"] == "route_search"
        )),
        "pair_counts_by_split": dict(Counter(row["model_split"] for row in pairs)),
        "seed_grade_counts": dict(Counter(str(row["seed_grade"]) for row in pairs)),
        "selection_role_counts": dict(Counter(row["selection_role"] for row in pairs)),
        "mean_pairs_per_subgoal": len(pairs) / max(len(subgoals), 1),
        "projected_pairs_for_351_tasks": (
            len(pairs) / max(len(tasks), 1) * 351
        ),
        "fixed_parent_memory_interface": True,
        "same_decomposer_for_train_and_inference": True,
        "judge_called": False,
        "official_validation_or_test_used": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
