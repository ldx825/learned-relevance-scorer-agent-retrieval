#!/usr/bin/env python3
"""Build compact train-only phase--operation supervision for TravelPlanner.

Unlike the typed-hard pilot, a phase query must retrieve several complementary
operation types.  Labels combine phase relevance with the executable quality
of each operation's train-derived text.  No official validation/test artifact
is read, and all candidates remain traceable to the fixed45-parent bank.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from travelplanner_atomic_memory import seed_grade


PHASES: tuple[dict[str, Any], ...] = (
    {
        "key": "transport_and_evidence",
        "goal": "collect complete feasible routes and preserve exact transport evidence",
        "anchor": ("route_search",),
        "required_complement": ("route_closure",),
        "primary": ("route_search", "route_closure"),
        "conditional": {"transportation": "transport_restriction"},
        "support": ("notebook_grounding", "final_constraint_audit", "repair_before_planner"),
    },
    {
        "key": "lodging_and_constraints",
        "goal": "select lodging that covers all nights, travelers, room requirements, and house rules",
        "anchor": ("lodging_search",),
        "required_complement": (),
        "primary": ("lodging_search", "minimum_nights", "party_capacity"),
        "conditional": {"room_type": "room_type", "house_rule": "house_rule"},
        "support": ("notebook_grounding", "full_party_budget", "final_constraint_audit", "repair_before_planner"),
    },
    {
        "key": "dining_and_cuisine",
        "goal": "cover every meal and requested cuisine without repeated or fabricated restaurants",
        "anchor": ("dining_search",),
        "required_complement": (),
        "primary": ("dining_search", "restaurant_uniqueness"),
        "conditional": {"cuisine": "cuisine_coverage"},
        "support": ("notebook_grounding", "full_party_budget", "final_constraint_audit", "repair_before_planner"),
    },
    {
        "key": "attractions_and_uniqueness",
        "goal": "select grounded attractions for each destination while avoiding repetition",
        "anchor": ("attraction_search",),
        "required_complement": (),
        "primary": ("attraction_search", "attraction_uniqueness"),
        "conditional": {},
        "support": ("notebook_grounding", "full_party_budget", "final_constraint_audit", "repair_before_planner"),
    },
    {
        "key": "budget_and_final_audit",
        "goal": "verify evidence, all visible constraints, complete party cost, and repair failures before finalization",
        "anchor": ("full_party_budget",),
        "required_complement": ("final_constraint_audit",),
        "primary": ("full_party_budget", "final_constraint_audit", "repair_before_planner"),
        "conditional": {},
        "support": (
            "notebook_grounding", "route_closure", "minimum_nights",
            "restaurant_uniqueness", "attraction_uniqueness",
        ),
    },
)


QUALITY_TERMS: dict[str, tuple[tuple[str, ...], ...]] = {
    "notebook_grounding": (
        ("notebook",), ("exact", "concrete", "specific"), ("name", "entity"),
        ("date", "time"), ("price", "cost", "rate"), ("final itinerary", "all trip components"),
    ),
    "route_closure": (
        ("outward", "origin departure", "departure"), ("inter-city", "every leg", "all route legs"),
        ("return",), ("date consistency", "time-consistent", "itinerary dates"),
        ("sandbox", "tool-returned", "notebook evidence"),
    ),
    "full_party_budget": (
        ("transport",), ("lodging", "accommodation"), ("dining", "meal"), ("attraction",),
        ("party", "people", "person"), ("per-person", "per-room", "per-night", "per-leg", "multiplier"),
        ("sum", "total cost", "calculate total"), ("budget",),
    ),
    "final_constraint_audit": (
        ("sandbox", "tool-returned", "supporting evidence", "notebook evidence"),
        ("route", "transport"), ("lodging", "accommodation", "minimum-night"),
        ("restaurant", "meal", "cuisine"), ("attraction",), ("budget", "cost"),
        ("requested constraint", "explicit constraint", "every visible"),
    ),
    "repair_before_planner": (
        ("re-search", "repeat the search", "search again"), ("failed check", "violation", "conflict"),
        ("constraint", "budget"), ("rather than invent", "rather than fabricate"),
        ("repeat the audit", "checks pass", "before planner"),
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def operation_text(operation: dict[str, Any]) -> str:
    return " ".join(str(operation.get(field, "")) for field in (
        "goal", "precondition", "procedure", "success_evidence", "failure_mode",
    )).lower()


def procedure_fingerprint(operation: dict[str, Any]) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(operation.get("procedure", "")).lower()))


def quality_score(operation: dict[str, Any]) -> float:
    text = operation_text(operation)
    procedure_tokens = re.findall(r"[a-z0-9]+", str(operation.get("procedure", "")).lower())
    success_tokens = re.findall(r"[a-z0-9]+", str(operation.get("success_evidence", "")).lower())
    failure_tokens = re.findall(r"[a-z0-9]+", str(operation.get("failure_mode", "")).lower())
    generic = (
        0.25 * min(len(procedure_tokens) / 18.0, 1.0)
        + 0.20 * min(len(success_tokens) / 12.0, 1.0)
        + 0.15 * min(len(failure_tokens) / 12.0, 1.0)
        + 0.20 * float(bool(operation.get("workflow_evidence")))
        + 0.20 * float(any(word in text for word in (
            "search", "check", "verify", "record", "store", "sum", "reject", "repeat", "use ",
        )))
    )
    checks = QUALITY_TERMS.get(operation["operation_key"])
    if not checks:
        return generic
    rubric = sum(any(term in text for term in alternatives) for alternatives in checks) / len(checks)
    return 0.70 * rubric + 0.30 * generic


def structural_distance(task: dict[str, Any], memory: dict[str, Any]) -> float:
    return (
        abs(int(task["days"]) - int(memory["days"])) / 7.0
        + abs(int(task["visiting_city_number"]) - int(memory["visiting_city_number"])) / 3.0
        + abs(int(task["people_number"]) - int(memory["people_number"])) / 10.0
        + abs(float(task["budget_per_person_day"]) - float(memory["budget_per_person_day"]))
        / max(float(task["budget_per_person_day"]), 1.0)
    )


def required_keys(phase: dict[str, Any], signature: dict[str, Any]) -> list[str]:
    keys = list(phase["primary"])
    keys.extend(operation for field, operation in phase["conditional"].items() if signature.get(field))
    return keys


def select_best_by_key(
    keys: list[str] | tuple[str, ...],
    operations_by_key: dict[str, list[dict[str, Any]]],
    signature: dict[str, Any],
    selected_ids: set[str],
    selected_procedures: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    chosen = []
    for key in keys:
        raw_candidates = []
        for row in operations_by_key.get(key, []):
            field = row.get("applicability_field")
            synthetic_subgoal = {
                "operation_key": key,
                "operation_family": row["operation_family"],
                "required_value": signature.get(field) if field else None,
            }
            grade, _ = seed_grade(synthetic_subgoal, row)
            if grade != 0:
                raw_candidates.append(row)
        if not raw_candidates:
            continue
        best_quality = max(quality_score(row) for row in raw_candidates)
        # Quality is a gate, not a popularity objective.  Once an operation is
        # sufficiently complete, prefer the structurally closest train Memory
        # so the same globally verbose template cannot monopolize Grade 2.
        candidates = sorted(
            (row for row in raw_candidates if quality_score(row) >= best_quality - 0.12),
            key=lambda row: (
                structural_distance(signature, row["source_signature"]),
                -quality_score(row), row["operation_id"],
            ),
        )
        candidate = next((row for row in candidates if (
            row["operation_id"] not in selected_ids
            and procedure_fingerprint(row) not in selected_procedures
        )), None)
        if candidate is None:
            continue
        selected_ids.add(candidate["operation_id"])
        selected_procedures.add(procedure_fingerprint(candidate))
        chosen.append(candidate)
        if len(chosen) == limit:
            break
    return chosen


def select_incomplete_relevant(
    keys: list[str] | tuple[str, ...],
    operations_by_key: dict[str, list[dict[str, Any]]],
    signature: dict[str, Any],
    selected_ids: set[str],
    selected_procedures: set[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Select structurally close but materially incomplete same-phase text."""
    pool = []
    for key in keys:
        values = operations_by_key.get(key, [])
        if not values:
            continue
        best_quality = max(quality_score(row) for row in values)
        for row in values:
            if quality_score(row) > best_quality - 0.20:
                continue
            pool.append(row)
    pool.sort(key=lambda row: (
        structural_distance(signature, row["source_signature"]),
        quality_score(row), row["operation_id"],
    ))
    chosen = []
    for row in pool:
        if row["operation_id"] in selected_ids or procedure_fingerprint(row) in selected_procedures:
            continue
        selected_ids.add(row["operation_id"])
        selected_procedures.add(procedure_fingerprint(row))
        chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgoals", type=Path, required=True)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-phase", type=int, default=10)
    args = parser.parse_args()

    subgoals = read_jsonl(args.subgoals)
    operations = read_jsonl(args.operations)
    if any(row.get("validation_or_test_used") is not False for row in subgoals + operations):
        raise ValueError("validation/test-derived input detected")
    families: dict[str, dict[str, Any]] = {}
    for row in subgoals:
        families.setdefault(row["family_id"], row)
    if len(families) != 396 or len(operations) != 519:
        raise ValueError("unexpected train-derived task or operation count")
    operations_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for operation in operations:
        operations_by_key[operation["operation_key"]].append(operation)

    phase_queries: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for family_id, source in sorted(families.items()):
        signature = source["signature"]
        for phase in PHASES:
            query_id = f"{family_id}:phase:{phase['key']}"
            query_text = (
                f"Full travel task: {source['task_query']}\n"
                f"Current planning phase: {phase['key']}.\n"
                f"Retrieve complementary procedural memory operations to {phase['goal']}."
            )
            phase_query = {
                "subgoal_id": query_id,
                "family_id": family_id,
                "model_split": source["model_split"],
                "task_query": source["task_query"],
                "subgoal_query": query_text,
                "operation_key": phase["key"],
                "operation_family": phase["key"],
                "phase_key": phase["key"],
                "required_operation_keys": required_keys(phase, signature),
                "support_operation_keys": list(phase["support"]),
                "signature": signature,
                "decomposer": "travelplanner_visible_phase_contract_v1",
                "validation_or_test_used": False,
            }
            phase_queries.append(phase_query)
            selected_ids: set[str] = set()
            selected_procedures: set[str] = set()

            primary = select_best_by_key(
                phase_query["required_operation_keys"], operations_by_key, signature,
                selected_ids, selected_procedures, limit=3,
            )
            support_keys = [
                key for key in phase_query["required_operation_keys"][3:] + phase_query["support_operation_keys"]
                if key not in {row["operation_key"] for row in primary}
            ]
            support = select_best_by_key(
                support_keys, operations_by_key, signature, selected_ids, selected_procedures, limit=3,
            )
            incomplete_primary = select_incomplete_relevant(
                phase_query["required_operation_keys"], operations_by_key, signature,
                selected_ids, selected_procedures, limit=1,
            )
            incomplete_support = select_incomplete_relevant(
                phase_query["support_operation_keys"], operations_by_key, signature,
                selected_ids, selected_procedures, limit=1,
            )

            unrelated = [
                operation for operation in operations
                if operation["operation_key"] not in set(
                    phase_query["required_operation_keys"] + phase_query["support_operation_keys"]
                )
            ]
            unrelated.sort(key=lambda row: (
                structural_distance(signature, row["source_signature"]),
                -quality_score(row), row["operation_id"],
            ))
            negatives = []
            for operation in unrelated:
                if operation["operation_id"] in selected_ids or procedure_fingerprint(operation) in selected_procedures:
                    continue
                selected_ids.add(operation["operation_id"])
                selected_procedures.add(procedure_fingerprint(operation))
                negatives.append(operation)
                if (
                    len(primary) + len(support) + len(incomplete_primary)
                    + len(incomplete_support) + len(negatives)
                    == args.pairs_per_phase
                ):
                    break
            if (
                len(primary) + len(support) + len(incomplete_primary)
                + len(incomplete_support) + len(negatives)
                != args.pairs_per_phase
            ):
                raise ValueError(f"cannot fill compact phase group: {query_id}")

            for grade, role, values in (
                (2, "phase_required_high_quality", primary),
                (1, "phase_complement_high_quality", support),
                (1, "phase_required_incomplete_hard_support", incomplete_primary),
                (0, "phase_complement_too_incomplete", incomplete_support),
                (0, "phase_unrelated_structural_hard_negative", negatives),
            ):
                for operation in values:
                    pairs.append({
                        "subgoal_id": query_id,
                        "family_id": family_id,
                        "model_split": source["model_split"],
                        "subgoal_query": query_text,
                        "phase_key": phase["key"],
                        "operation_id": operation["operation_id"],
                        "operation_key": operation["operation_key"],
                        "parent_memory_id": int(operation["parent_memory_id"]),
                        "operation_text": operation["operation_text"],
                        "final_grade": grade,
                        "selection_role": role,
                        "operation_quality_score": quality_score(operation),
                        "label_source": "deterministic_train_only_phase_quality_v1",
                        "judge_required": False,
                        "validation_or_test_used": False,
                    })

    train = [row for row in pairs if row["model_split"] == "internal_train"]
    dev = [row for row in pairs if row["model_split"] == "internal_dev"]
    train_families = {row["family_id"] for row in train}
    dev_families = {row["family_id"] for row in dev}
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_query[row["subgoal_id"]].append(row)
    report = {
        "schema_version": "memp.travelplanner.atomic_phase_pairs.v1",
        "parent_memory_count": 45,
        "operation_count": len(operations),
        "train_derived_task_count": len(families),
        "phase_count_per_task": len(PHASES),
        "phase_query_count": len(phase_queries),
        "pair_count": len(pairs),
        "train_pair_count": len(train),
        "internal_dev_pair_count": len(dev),
        "grade_counts_by_split": {
            "internal_train": dict(Counter(str(row["final_grade"]) for row in train)),
            "internal_dev": dict(Counter(str(row["final_grade"]) for row in dev)),
        },
        "phase_query_counts": dict(Counter(row["phase_key"] for row in phase_queries)),
        "operation_key_counts_by_grade": {
            str(grade): dict(Counter(row["operation_key"] for row in pairs if row["final_grade"] == grade))
            for grade in (2, 1, 0)
        },
        "group_size_counts": dict(Counter(str(len(rows)) for rows in by_query.values())),
        "groups_without_grade2": sum(not any(row["final_grade"] == 2 for row in rows) for rows in by_query.values()),
        "groups_without_grade0": sum(not any(row["final_grade"] == 0 for row in rows) for rows in by_query.values()),
        "duplicate_pair_count": len(pairs) - len({(row["subgoal_id"], row["operation_id"]) for row in pairs}),
        "train_family_count": len(train_families),
        "internal_dev_family_count": len(dev_families),
        "cross_split_family_count": len(train_families & dev_families),
        "query_and_label_source": "public TravelPlanner train-derived only",
        "validation_or_test_used": False,
        "judge_called": False,
    }
    report["audit_pass"] = all((
        report["pair_count"] == 19800,
        report["train_pair_count"] == 15850,
        report["internal_dev_pair_count"] == 3950,
        report["groups_without_grade2"] == 0,
        report["groups_without_grade0"] == 0,
        report["duplicate_pair_count"] == 0,
        report["cross_split_family_count"] == 0,
        report["validation_or_test_used"] is False,
    ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "phase_queries.jsonl", phase_queries)
    write_jsonl(args.output_dir / "train_pairs_resolved.jsonl", train)
    write_jsonl(args.output_dir / "internal_dev_pairs_resolved.jsonl", dev)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["audit_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
