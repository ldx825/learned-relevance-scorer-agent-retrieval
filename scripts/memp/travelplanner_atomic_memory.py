#!/usr/bin/env python3
"""Shared atomic-Memory contract for TravelPlanner training and inference.

The public MemP interface still owns a fixed parent Memory bank.  This module
only creates internal operation views, decomposes a visible task into matching
subgoals, and aggregates operation scores back to the original parent IDs.
It deliberately does not read validation/test labels, trajectories, or rewards.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Iterable


OPERATION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "route_search",
        "family": "route",
        "order": 10,
        "goal": "Find feasible outward, inter-city, and return transportation legs.",
        "precondition": "The origin, destination region, dates, and number of cities are known.",
        "procedure": "Search every required leg and retain exact route, time, and price evidence.",
        "success_evidence": "Every itinerary transition has a feasible searched transport option.",
        "failure_mode": "A route leg is omitted or replaced by an invented connection.",
    },
    {
        "key": "route_closure",
        "family": "route",
        "order": 20,
        "goal": "Verify that all route legs form a complete and time-consistent journey.",
        "precondition": "Candidate transport evidence has been collected for every leg.",
        "procedure": "Check origin departure, inter-city order, final return, and date consistency.",
        "success_evidence": "The complete route is closed and every leg fits the itinerary dates.",
        "failure_mode": "The plan lacks the outward, inter-city, or return leg.",
    },
    {
        "key": "transport_restriction",
        "family": "route",
        "order": 30,
        "conditional_field": "transportation",
        "goal": "Enforce the user's explicit transportation restriction.",
        "precondition": "The task explicitly requires or prohibits a transportation mode.",
        "procedure": "Reject conflicting legs and re-search until every leg obeys the restriction.",
        "success_evidence": "All selected legs use transportation allowed by the request.",
        "failure_mode": "A cheap or convenient leg violates the explicit transportation rule.",
    },
    {
        "key": "notebook_grounding",
        "family": "evidence",
        "order": 40,
        "goal": "Ground the final itinerary in exact tool-returned evidence.",
        "precondition": "A route, lodging, restaurant, or attraction candidate was selected.",
        "procedure": "Write exact names, cities, dates, times, and prices to the Notebook.",
        "success_evidence": "Every final itinerary item is supported by a Notebook record.",
        "failure_mode": "The final plan contains an entity or value that was never returned by a tool.",
    },
    {
        "key": "lodging_search",
        "family": "lodging",
        "order": 50,
        "goal": "Find lodging for every required night and destination city.",
        "precondition": "Travel dates, cities, party size, and lodging requirements are known.",
        "procedure": "Search lodging and retain city, dates, capacity, room type, rules, and price.",
        "success_evidence": "Every night has a feasible, tool-returned accommodation.",
        "failure_mode": "One or more nights/cities have no valid accommodation.",
    },
    {
        "key": "minimum_nights",
        "family": "lodging",
        "order": 60,
        "goal": "Validate accommodation minimum-night requirements.",
        "precondition": "A lodging candidate and stay duration are known.",
        "procedure": "Compare the planned stay against the property's minimum-night rule.",
        "success_evidence": "The stay length satisfies the selected property's rule.",
        "failure_mode": "The selected property requires more nights than the itinerary provides.",
    },
    {
        "key": "party_capacity",
        "family": "lodging",
        "order": 70,
        "goal": "Ensure lodging capacity covers the complete travel party.",
        "precondition": "Party size and property capacity are available.",
        "procedure": "Check capacity and compute the number of rooms/properties required.",
        "success_evidence": "All travelers are accommodated for every night.",
        "failure_mode": "A single-person or undersized listing is used for the full party.",
    },
    {
        "key": "room_type",
        "family": "lodging",
        "order": 80,
        "conditional_field": "room_type",
        "goal": "Enforce the explicitly requested accommodation room type.",
        "precondition": "The task contains a room-type requirement.",
        "procedure": "Reject lodging with a different room type and re-search.",
        "success_evidence": "Every selected accommodation has the requested room type.",
        "failure_mode": "A private/shared room is substituted for an entire room, or vice versa.",
    },
    {
        "key": "house_rule",
        "family": "lodging",
        "order": 90,
        "conditional_field": "house_rule",
        "goal": "Enforce the requested accommodation house rule.",
        "precondition": "The task contains an explicit house-rule requirement.",
        "procedure": "Check the property's rules and reject incompatible accommodation.",
        "success_evidence": "Every selected property satisfies the requested house rule.",
        "failure_mode": "The accommodation conflicts with the smoking, pets, visitors, or party rule.",
    },
    {
        "key": "dining_search",
        "family": "dining",
        "order": 100,
        "goal": "Select valid restaurants for all required meals.",
        "precondition": "The itinerary cities, dates, and meal slots are known.",
        "procedure": "Search restaurants in the correct city and assign breakfast, lunch, and dinner.",
        "success_evidence": "Every required meal has a valid restaurant in the correct city.",
        "failure_mode": "A meal is missing or assigned to a restaurant in the wrong city.",
    },
    {
        "key": "cuisine_coverage",
        "family": "dining",
        "order": 110,
        "conditional_field": "cuisine",
        "goal": "Cover every cuisine explicitly requested by the user.",
        "precondition": "The task includes one or more requested cuisines.",
        "procedure": "Choose tool-returned restaurants whose cuisines jointly cover the request.",
        "success_evidence": "Every requested cuisine appears in at least one selected restaurant.",
        "failure_mode": "A requested cuisine is silently omitted from the itinerary.",
    },
    {
        "key": "restaurant_uniqueness",
        "family": "dining",
        "order": 120,
        "goal": "Avoid repeating restaurants across meal slots.",
        "precondition": "Restaurant candidates have been assigned to meals.",
        "procedure": "Deduplicate restaurant names and re-search repeated assignments.",
        "success_evidence": "No restaurant is repeated in the final itinerary.",
        "failure_mode": "The same restaurant is reused across days or meal slots.",
    },
    {
        "key": "attraction_search",
        "family": "attraction",
        "order": 130,
        "goal": "Select valid attractions for each destination and itinerary day.",
        "precondition": "Destination cities and available day slots are known.",
        "procedure": "Search attractions in each city and retain exact names and locations.",
        "success_evidence": "Each required day/city has a valid attraction assignment.",
        "failure_mode": "An attraction is missing, fabricated, or located in the wrong city.",
    },
    {
        "key": "attraction_uniqueness",
        "family": "attraction",
        "order": 140,
        "goal": "Avoid repeating attractions across itinerary days.",
        "precondition": "Attractions have been assigned to itinerary days.",
        "procedure": "Deduplicate attraction names and replace repeated selections.",
        "success_evidence": "Every attraction in the itinerary is unique.",
        "failure_mode": "The same attraction is scheduled multiple times.",
    },
    {
        "key": "full_party_budget",
        "family": "budget",
        "order": 150,
        "goal": "Account for the complete trip cost for the entire party.",
        "precondition": "Transport, lodging, dining, attraction prices, party size, and duration are known.",
        "procedure": "Sum all costs with the correct per-person, per-room, per-night, and per-leg multipliers.",
        "success_evidence": "The complete party-level total is within the user's budget.",
        "failure_mode": "Only a single-person or single-day cost is compared with the total budget.",
    },
    {
        "key": "final_constraint_audit",
        "family": "verification",
        "order": 160,
        "goal": "Audit every explicit and structural constraint before finalization.",
        "precondition": "A complete candidate itinerary and Notebook evidence are available.",
        "procedure": "Check route, transport, lodging, meals, cuisines, attractions, uniqueness, and budget.",
        "success_evidence": "Every visible task requirement passes with supporting evidence.",
        "failure_mode": "Planner is called while a required field or constraint remains unchecked.",
    },
    {
        "key": "repair_before_planner",
        "family": "verification",
        "order": 170,
        "goal": "Repair failed checks before invoking the final Planner.",
        "precondition": "At least one constraint, evidence, or budget check failed.",
        "procedure": "Re-search the failing component, update evidence, and repeat the audit.",
        "success_evidence": "No known violation remains when Planner is called.",
        "failure_mode": "The system fabricates a workaround or finalizes a known-invalid itinerary.",
    },
)

SPEC_BY_KEY = {row["key"]: row for row in OPERATION_SPECS}

OPERATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "route_search": ("flightsearch", "googledistancematrix", "transportation options"),
    "route_closure": ("route closure", "transportation consistency", "every leg"),
    "transport_restriction": ("transportation restriction", "transport restriction"),
    "notebook_grounding": ("notebookwrite", "notebook evidence", "in the notebook"),
    "lodging_search": ("accommodationsearch", "for accommodations", "lodging"),
    "minimum_nights": ("minimum-night", "minimum night"),
    "party_capacity": ("party size", "capacity"),
    "room_type": ("room type",),
    "house_rule": ("house rule",),
    "dining_search": ("restaurantsearch", "breakfast", "lunch", "dinner"),
    "cuisine_coverage": ("cuisine preferences", "requested cuisine"),
    "restaurant_uniqueness": ("restaurant is repeated", "non-repetition of restaurants"),
    "attraction_search": ("attractionsearch", "select unique attractions"),
    "attraction_uniqueness": ("unique attractions", "non-repetition of attractions"),
    "full_party_budget": ("total cost", "party size", "budget"),
    "final_constraint_audit": ("before running planner", "verify all", "requested constraints"),
    "repair_before_planner": ("re-search", "rather than fabricate"),
}

CAPABILITY_EVIDENCE_KEY = {
    "full_party_budget": "full_party_cost",
    "final_constraint_audit": "explicit_constraints",
}


def _workflow_evidence(profile: dict[str, Any], operation_key: str) -> str:
    """Return the shortest operation-specific evidence actually present in a Memory."""
    evidence_key = CAPABILITY_EVIDENCE_KEY.get(operation_key, operation_key)
    extracted = profile.get("capability_evidence", {}).get(evidence_key) or []
    clean_extracted = [" ".join(str(value).split()) for value in extracted if value]
    if clean_extracted:
        return min(clean_extracted, key=len)
    workflow = " ".join(str(profile.get("workflow", "")).split())
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", workflow) if part.strip()]
    keywords = OPERATION_KEYWORDS.get(operation_key, ())
    matched = [sentence for sentence in sentences if any(word in sentence.lower() for word in keywords)]
    return min(matched, key=len) if matched else ""


def _operation_text(
    spec: dict[str, Any], applicability: Any = None, source_evidence: str = ""
) -> str:
    lines = [
        f"Operation: {spec['key']}",
        f"Goal: {spec['goal']}",
        f"Precondition: {spec['precondition']}",
        f"Procedure: {spec['procedure']}",
        f"Success evidence: {spec['success_evidence']}",
        f"Failure mode: {spec['failure_mode']}",
    ]
    if applicability:
        lines.insert(2, f"Source applicability: {applicability}")
    if source_evidence:
        lines.append(f"Grounding from parent Memory: {source_evidence}")
    return "\n".join(lines)


def decompose_memory(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Create grounded operation views while retaining the parent Memory ID."""
    signature = profile["source_signature"]
    capabilities = profile.get("capabilities", {})
    rows: list[dict[str, Any]] = []
    for spec in OPERATION_SPECS:
        field = spec.get("conditional_field")
        applicability = signature.get(field) if field else None
        if field and not applicability:
            continue
        capability_key = {
            "full_party_budget": "full_party_cost",
            "final_constraint_audit": "explicit_constraints",
        }.get(spec["key"], spec["key"])
        # A false extracted capability is stronger evidence than a generic
        # workflow phrase; do not invent that operation for this Memory.
        if capability_key in capabilities and not capabilities[capability_key]:
            continue
        operation_id = f"memory_{int(profile['memory_id']):03d}:{spec['key']}"
        source_evidence = _workflow_evidence(profile, spec["key"])
        rows.append(
            {
                "operation_id": operation_id,
                "parent_memory_id": int(profile["memory_id"]),
                "parent_source": profile["source"],
                "parent_memory_query": profile["memory_query"],
                "operation_key": spec["key"],
                "operation_family": spec["family"],
                "order": spec["order"],
                "goal": spec["goal"],
                "precondition": spec["precondition"],
                "procedure": spec["procedure"],
                "success_evidence": spec["success_evidence"],
                "failure_mode": spec["failure_mode"],
                "applicability_field": field,
                "applicability_value": applicability,
                "source_workflow_evidence": source_evidence,
                "source_signature": signature,
                "operation_text": _operation_text(spec, applicability, source_evidence),
                "extraction": "deterministic_from_train_memory_profile",
                "validation_or_test_used": False,
            }
        )
    return rows


def decompose_task(
    task_query: str,
    signature: dict[str, Any],
    *,
    family_id: str,
    model_split: str,
) -> list[dict[str, Any]]:
    """Decompose only user-visible fields; used identically at train/test time."""
    rows: list[dict[str, Any]] = []
    for spec in OPERATION_SPECS:
        field = spec.get("conditional_field")
        value = signature.get(field) if field else None
        if field and not value:
            continue
        subgoal_id = f"{family_id}:subgoal:{spec['key']}"
        text = (
            f"Full travel task: {task_query}\n"
            f"Current procedural requirement: {spec['goal']}"
        )
        if value:
            text += f"\nExplicit requirement value: {value}"
        rows.append(
            {
                "subgoal_id": subgoal_id,
                "family_id": family_id,
                "model_split": model_split,
                "task_query": task_query,
                "subgoal_query": text,
                "operation_key": spec["key"],
                "operation_family": spec["family"],
                "order": spec["order"],
                "required_value": value,
                "signature": signature,
                "decomposer": "travelplanner_visible_contract_v1",
                "validation_or_test_used": False,
            }
        )
    return rows


TARGET_EVIDENCE_TERMS: dict[str, tuple[str, ...]] = {
    "route_search": ("flight", "transportation", "route"),
    "route_closure": ("route closure", "return", "every leg"),
    "transport_restriction": ("transport", "no flight", "no self-driving"),
    "notebook_grounding": ("notebook", "record", "store", "log"),
    "lodging_search": ("accommodation", "lodging"),
    "minimum_nights": ("minimum-night", "minimum night"),
    "party_capacity": ("capacity", "party size", "all travelers"),
    "room_type": ("room type", "entire room", "private room"),
    "house_rule": ("house rule", "smoking", "pets", "parties", "visitors", "children under 10"),
    "dining_search": ("restaurant", "meal"),
    "cuisine_coverage": ("cuisine",),
    "restaurant_uniqueness": ("repeated restaurant", "non-repeated", "non-repeating"),
    "attraction_search": ("attraction",),
    "attraction_uniqueness": ("unique attraction", "repeated attraction", "no duplicates"),
    "full_party_budget": ("total cost", "party size", "full party", "budget"),
    "final_constraint_audit": ("verify", "validate", "confirm", "audit"),
    "repair_before_planner": ("re-search", "alternative", "repair"),
}


def _normalized_values(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).strip().lower() for item in values if item is not None]


def seed_grade(subgoal: dict[str, Any], operation: dict[str, Any]) -> tuple[int | str, str]:
    """High-precision seed only; a grouped Judge may refine ambiguous pairs."""
    if subgoal["operation_key"] == operation["operation_key"]:
        required = subgoal.get("required_value")
        offered = operation.get("applicability_value")
        if required and offered != required:
            evidence = " ".join(
                str(value) for value in (
                    operation.get("operation_text"),
                    operation.get("workflow_evidence"),
                    operation.get("source_workflow_evidence"),
                ) if value
            ).lower()
            offered_values = _normalized_values(offered)
            required_values = _normalized_values(required)
            explicitly_specialized = (
                bool(offered_values)
                and all(value in evidence for value in offered_values)
                and set(offered_values) != set(required_values)
            )
            if explicitly_specialized:
                return 0, "same operation but parent evidence is specialized to a conflicting value"
            return 2, "direct value-agnostic operation transfers to the requested value"
        return 2, "direct operation match with compatible visible applicability"
    if subgoal["operation_family"] == operation["operation_family"]:
        evidence = str(operation.get("operation_text", "")).lower()
        terms = TARGET_EVIDENCE_TERMS[subgoal["operation_key"]]
        if any(term in evidence for term in terms):
            return 1, "same workflow family explicitly supports the target requirement"
        return "uncertain", "same workflow family but target-specific support is not visible"
    verification_support = (
        operation["operation_family"] == "verification"
        and subgoal["operation_family"] not in {"evidence", "verification"}
    )
    if verification_support:
        return 1, "cross-cutting evidence or verification support"
    return 0, "different operation family without direct procedural support"


def aggregate_parent_scores(
    scored_operations: Iterable[dict[str, Any]],
    *,
    subgoal_count: int,
) -> list[dict[str, Any]]:
    """Aggregate operation scores back to the fixed parent-Memory interface."""
    by_parent: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in scored_operations:
        by_parent[int(row["parent_memory_id"])][row["subgoal_id"]].append(float(row["score"]))
    ranked: list[dict[str, Any]] = []
    for parent_id, per_subgoal in by_parent.items():
        best = {key: max(values) for key, values in per_subgoal.items()}
        mean_score = sum(best.values()) / max(subgoal_count, 1)
        covered = sum(score > 0 for score in best.values())
        ranked.append(
            {
                "parent_memory_id": parent_id,
                "score": mean_score,
                "covered_subgoals": covered,
                "subgoal_count": subgoal_count,
                "coverage": covered / max(subgoal_count, 1),
                "best_operation_score_by_subgoal": best,
            }
        )
    return sorted(ranked, key=lambda row: (-row["score"], -row["coverage"], row["parent_memory_id"]))


def render_dynamic_memory(
    task_query: str,
    selected_operations: Iterable[dict[str, Any]],
) -> str:
    """Render deduplicated operations without discarding parent Memory evidence."""
    best_by_key: dict[str, dict[str, Any]] = {}
    for row in selected_operations:
        key = row["operation_key"]
        if key not in best_by_key or float(row.get("score", 0)) > float(best_by_key[key].get("score", 0)):
            best_by_key[key] = row
    ordered = sorted(best_by_key.values(), key=lambda row: (int(row["order"]), row["operation_key"]))
    lines = [
        "# Task-specific procedural memory",
        "",
        f"Task: {task_query}",
        "",
        (
            "Execution contract: carry out the applicable operations below, preserve exact "
            "tool-returned evidence in Notebook, and do not call Planner while a required "
            "success check is false. If a check fails, repair the failing component first."
        ),
        "",
    ]
    rendered_evidence: set[str] = set()
    for index, row in enumerate(ordered, 1):
        evidence = row.get("workflow_evidence") or row.get("source_workflow_evidence")
        lines.extend([
            f"{index}. Operation: {row['operation_key']}",
            f"   Action: {row['procedure']}",
            f"   Required success check: {row['success_evidence']}",
            f"   Failure to prevent: {row['failure_mode']}",
        ])
        if row.get("applicability_value"):
            lines.append(f"   Current applicability constraint: {row['applicability_value']}")
        normalized_evidence = " ".join(str(evidence or "").split())
        if normalized_evidence and normalized_evidence not in rendered_evidence:
            rendered_evidence.add(normalized_evidence)
            lines.append(
                "   Reusable evidence from a training Memory: "
                f"{normalized_evidence} Transfer only the procedural pattern. Ignore source cities, "
                "dates, entities, prices, and constraints unless they are explicitly present in the "
                "current task."
            )
    return "\n".join(lines)
