#!/usr/bin/env python3
"""Build train-only atomic task--Memory supervision for TravelPlanner.

This is a data-only extension of the ALFWorld Memory V17 contract.  It derives
fine-grained retrieval queries from the 45 public train tasks and labels the
45-memory bank from auditable workflow capability evidence.  It never reads
official validation/test tasks, trajectories, rewards, or gold plans.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALWAYS_CAPABILITIES = (
    "route_closure", "transport_consistency", "sandbox_evidence",
    "notebook_grounding", "no_fabrication", "lodging_feasibility",
    "minimum_nights", "party_capacity", "dining_coverage",
    "restaurant_uniqueness", "attraction_coverage", "attraction_uniqueness",
    "non_repetition", "budget_accounting", "full_party_cost",
    "repair_before_planner",
)

CAPABILITY_TEXT = {
    "route_closure": "complete every outward, inter-city, and return route leg",
    "transport_consistency": "keep one feasible and non-conflicting transport mode for every leg",
    "sandbox_evidence": "use only entities and fields returned by the available tools",
    "notebook_grounding": "record exact selected entities and costs in the notebook before planning",
    "no_fabrication": "re-search instead of inventing or silently changing an entity",
    "lodging_feasibility": "verify lodging city, dates, capacity, price, rules, and minimum stay",
    "minimum_nights": "satisfy every accommodation minimum-night requirement",
    "party_capacity": "verify that lodging and costs cover the complete party",
    "dining_coverage": "cover all required meals and cuisines with valid restaurants",
    "restaurant_uniqueness": "avoid repeating a restaurant across the itinerary",
    "cuisine_coverage": "cover every explicitly requested cuisine",
    "attraction_coverage": "select enough valid attractions in the correct cities",
    "attraction_uniqueness": "avoid repeating an attraction across itinerary days",
    "non_repetition": "prevent repeated restaurants and attractions",
    "house_rule": "satisfy the requested accommodation house rule",
    "room_type": "satisfy the requested accommodation room type",
    "transport_restriction": "satisfy the explicit transportation restriction",
    "explicit_constraints": "check every explicit transport, lodging, and cuisine constraint",
    "budget_accounting": "sum every transport, lodging, meal, and attraction cost",
    "full_party_cost": "apply prices to the complete party and all itinerary days",
    "repair_before_planner": "repair every failed check before calling the final planner",
}

PARENT_VIEW = {
    "route_closure": "route_closure",
    "transport_consistency": "route_closure",
    "sandbox_evidence": "sandbox_evidence",
    "notebook_grounding": "sandbox_evidence",
    "no_fabrication": "sandbox_evidence",
    "lodging_feasibility": "lodging_feasibility",
    "minimum_nights": "lodging_feasibility",
    "party_capacity": "lodging_feasibility",
    "dining_coverage": "dining_coverage",
    "restaurant_uniqueness": "dining_coverage",
    "cuisine_coverage": "dining_coverage",
    "attraction_coverage": "attraction_coverage",
    "attraction_uniqueness": "attraction_coverage",
    "non_repetition": "final_verification",
    "house_rule": "explicit_constraints",
    "room_type": "explicit_constraints",
    "transport_restriction": "explicit_constraints",
    "explicit_constraints": "explicit_constraints",
    "budget_accounting": "budget_accounting",
    "full_party_cost": "budget_accounting",
    "repair_before_planner": "final_verification",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def active_capabilities(signature: dict[str, Any]) -> list[str]:
    result = list(ALWAYS_CAPABILITIES)
    if signature.get("cuisine"):
        result.append("cuisine_coverage")
    if signature.get("house_rule"):
        result.append("house_rule")
    if signature.get("room_type"):
        result.append("room_type")
    if signature.get("transportation"):
        result.append("transport_restriction")
    if int(signature.get("active_constraint_count", 0)):
        result.append("explicit_constraints")
    return result


def support_strength(profile: dict[str, Any], capability: str) -> int:
    flags = profile["capabilities"]
    if capability in flags:
        return 2 if flags[capability] else 0
    components = {
        "lodging_feasibility": ("minimum_nights", "party_capacity"),
        "dining_coverage": ("restaurant_uniqueness", "cuisine_coverage"),
        "attraction_coverage": ("attraction_uniqueness",),
        "non_repetition": ("restaurant_uniqueness", "attraction_uniqueness"),
        "explicit_constraints": (
            "house_rule", "room_type", "transport_restriction", "cuisine_coverage"
        ),
    }[capability]
    count = sum(bool(flags[name]) for name in components)
    if count == len(components):
        return 2
    return 1 if count else 0


def exact_constraint_matches(task: dict[str, Any], memory: dict[str, Any]) -> int:
    return sum(
        task.get(key) is not None and task.get(key) == memory.get(key)
        for key in ("house_rule", "room_type", "transportation")
    ) + int(
        bool(task.get("cuisine"))
        and bool(set(task["cuisine"]) & set(memory.get("cuisine") or []))
    )


def structural_distance(task: dict[str, Any], memory: dict[str, Any]) -> float:
    return (
        abs(int(task["days"]) - int(memory["days"])) / 7.0
        + abs(int(task["visiting_city_number"]) - int(memory["visiting_city_number"])) / 3.0
        + abs(int(task["people_number"]) - int(memory["people_number"])) / 10.0
        + abs(float(task["budget_per_person_day"]) - float(memory["budget_per_person_day"]))
        / max(float(task["budget_per_person_day"]), 1.0)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-views", type=Path, required=True)
    parser.add_argument("--memory-profiles", type=Path, required=True)
    parser.add_argument("--parent-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    views = read_jsonl(args.query_views)
    profiles = json.loads(args.memory_profiles.read_text(encoding="utf-8"))
    parent_labels = read_jsonl(args.parent_labels)
    full_by_family = {row["family_id"]: row for row in views if row["view_type"] == "full_task"}
    if len(full_by_family) != 45 or len(profiles) != 45:
        raise ValueError("expected45 public-train families and45 train-derived Memories")
    if any(row.get("validation_or_test_used") is not False for row in views + profiles + parent_labels):
        raise ValueError("official validation/test-derived input detected")

    view_task_id = {row["view_id"]: index for index, row in enumerate(views)}
    parent_label_by_pair = {
        (int(row["task_id"]), int(row["memory_id"])): row for row in parent_labels
    }

    queries: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for family_id, parent in sorted(full_by_family.items()):
        signature = parent["signature"]
        for capability in active_capabilities(signature):
            parent_view_id = f"{family_id}:{PARENT_VIEW[capability]}"
            parent_task_id = view_task_id[parent_view_id]
            inherited_rows = [
                parent_label_by_pair[(parent_task_id, int(profile["memory_id"]))]
                for profile in profiles
            ]
            if not any(row["grade"] == 2 for row in inherited_rows):
                continue
            query_id = len(queries)
            query_text = (
                f"For this travel-planning task: {parent['task_query']}\n"
                f"Retrieve reusable procedural Memory specifically for this requirement: "
                f"{CAPABILITY_TEXT[capability]}."
            )
            queries.append(
                {
                    "task_id": query_id,
                    "view_id": f"{family_id}:atomic:{capability}",
                    "family_id": family_id,
                    "parent_train_index": parent["parent_train_index"],
                    "model_split": parent["model_split"],
                    "view_type": f"atomic_{capability}",
                    "task_query": parent["task_query"],
                    "query_text": query_text,
                    "required_capabilities": [capability],
                    "signature": signature,
                    "source": "public_train_deterministic_atomic_capability_view",
                    "validation_or_test_used": False,
                }
            )

            for profile in profiles:
                memory_id = int(profile["memory_id"])
                strength = support_strength(profile, capability)
                distance = structural_distance(signature, profile["source_signature"])
                exact = exact_constraint_matches(signature, profile["source_signature"])
                inherited = parent_label_by_pair[(parent_task_id, memory_id)]["grade"]
                if inherited == 2:
                    grade: int | str = 2
                    reason = "inherited audited parent-stage primary"
                elif inherited == 1 and strength > 0:
                    grade = 1
                    reason = "inherited parent support with atomic capability evidence"
                elif inherited in {1, "uncertain"} or strength > 0:
                    grade = "uncertain"
                    reason = "positive-unlabeled candidate without aligned parent-stage support"
                else:
                    grade = 0
                    reason = "inherited parent negative without atomic capability evidence"
                labels.append(
                    {
                        "task_id": query_id,
                        "view_id": queries[-1]["view_id"],
                        "family_id": family_id,
                        "internal_split": parent["model_split"],
                        "view_type": queries[-1]["view_type"],
                        "memory_id": memory_id,
                        "grade": grade,
                        "capability": capability,
                        "parent_view_id": parent_view_id,
                        "parent_task_id": parent_task_id,
                        "parent_grade": inherited,
                        "capability_support_strength": strength,
                        "exact_constraint_matches": exact,
                        "structural_distance": distance,
                        "calibration": reason,
                        "validation_or_test_used": False,
                    }
                )

    write_jsonl(args.output_dir / "query_views.jsonl", queries)
    write_jsonl(args.output_dir / "labels.jsonl", labels)
    report = {
        "schema_version": "memp.travelplanner.atomic_capability_data.v1",
        "parent_train_task_count": 45,
        "query_count": len(queries),
        "raw_pair_count": len(labels),
        "query_counts_by_split": dict(Counter(row["model_split"] for row in queries)),
        "query_counts_by_capability": dict(Counter(row["required_capabilities"][0] for row in queries)),
        "grade_counts": dict(Counter(str(row["grade"]) for row in labels)),
        "one_grade2_per_query": all(
            sum(row["task_id"] == task_id and row["grade"] == 2 for row in labels) == 1
            for task_id in range(len(queries))
        ),
        "family_split_disjoint": all(
            len({row["model_split"] for row in queries if row["family_id"] == family}) == 1
            for family in full_by_family
        ),
        "model_or_fusion_changed": False,
        "label_alignment": "atomic Grade-2/1 inherited from audited parent-stage labels",
        "validation_or_test_used": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
