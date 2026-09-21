#!/usr/bin/env python3

from travelplanner_atomic_memory import (
    aggregate_parent_scores,
    decompose_memory,
    decompose_task,
    render_dynamic_memory,
    seed_grade,
)
from build_travelplanner_atomic_memory_pilot import select_informative_operations
from build_travelplanner_atomic_phase_pairs import PHASES
from prepare_travelplanner_phase_dynamic_retrieval import compatible, select_role_complementary


def signature(**overrides):
    base = {
        "days": 5,
        "visiting_city_number": 2,
        "people_number": 4,
        "budget": 8000,
        "budget_per_person_day": 400.0,
        "transportation": "no flight",
        "room_type": "entire room",
        "house_rule": "smoking",
        "cuisine": ["Chinese"],
        "active_constraint_count": 4,
    }
    base.update(overrides)
    return base


def profile(memory_id=0, **overrides):
    workflow = overrides.pop(
        "_workflow",
        "Use FlightSearch for transportation options. Before running Planner, verify all constraints and re-search rather than fabricate.",
    )
    return {
        "memory_id": memory_id,
        "source": f"travelplanner_train_{memory_id:03d}",
        "memory_query": "Plan a constrained trip.",
        "workflow": workflow,
        "source_signature": signature(**overrides),
        "capabilities": {},
        "validation_or_test_used": False,
    }


def test_conditional_operations_follow_visible_constraints():
    rows = decompose_memory(profile())
    keys = {row["operation_key"] for row in rows}
    assert {"transport_restriction", "room_type", "house_rule", "cuisine_coverage"} <= keys
    plain = decompose_memory(
        profile(transportation=None, room_type=None, house_rule=None, cuisine=None)
    )
    plain_keys = {row["operation_key"] for row in plain}
    assert not {"transport_restriction", "room_type", "house_rule", "cuisine_coverage"} & plain_keys


def test_train_and_inference_task_decomposition_is_deterministic():
    first = decompose_task("Plan my trip", signature(), family_id="f0", model_split="train")
    second = decompose_task("Plan my trip", signature(), family_id="f0", model_split="train")
    assert first == second
    assert any(row["operation_key"] == "transport_restriction" for row in first)


def test_seed_grade_detects_exact_match_and_conflict():
    subgoal = next(
        row
        for row in decompose_task("Plan", signature(), family_id="f0", model_split="train")
        if row["operation_key"] == "transport_restriction"
    )
    exact = next(
        row for row in decompose_memory(profile())
        if row["operation_key"] == "transport_restriction"
    )
    conflict = next(
        row for row in decompose_memory(profile(
            1,
            transportation="no self-driving",
            _workflow="Enforce the transportation restriction no self-driving and verify it before Planner.",
        ))
        if row["operation_key"] == "transport_restriction"
    )
    assert seed_grade(subgoal, exact)[0] == 2
    assert seed_grade(subgoal, conflict)[0] == 0


def test_aggregation_keeps_parent_memory_ids():
    ranked = aggregate_parent_scores(
        [
            {"parent_memory_id": 3, "subgoal_id": "a", "score": 0.8},
            {"parent_memory_id": 3, "subgoal_id": "b", "score": 0.7},
            {"parent_memory_id": 4, "subgoal_id": "a", "score": 0.9},
        ],
        subgoal_count=2,
    )
    assert ranked[0]["parent_memory_id"] == 3
    assert {row["parent_memory_id"] for row in ranked} == {3, 4}


def test_dynamic_memory_deduplicates_operations():
    operation = decompose_memory(profile())[0]
    text = render_dynamic_memory("Plan my trip", [operation, dict(operation, score=1.0)])
    assert text.count(operation["procedure"]) == 1
    assert operation["success_evidence"] in text
    assert operation["failure_mode"] in text
    assert operation["source_workflow_evidence"] in text
    assert "do not call Planner" in text


def test_operation_text_is_grounded_in_parent_workflow():
    operation = next(
        row for row in decompose_memory(profile()) if row["operation_key"] == "route_search"
    )
    assert "FlightSearch" in operation["source_workflow_evidence"]
    assert "Grounding from parent Memory" in operation["operation_text"]


def test_compact_selection_does_not_recreate_cartesian_product():
    subgoal = next(
        row
        for row in decompose_task("Plan", signature(), family_id="f0", model_split="train")
        if row["operation_key"] == "transport_restriction"
    )
    operations = decompose_memory(profile()) + decompose_memory(
        profile(
            1,
            transportation="no self-driving",
            _workflow="Enforce the transportation restriction no self-driving and verify it before Planner.",
        )
    )
    selected = select_informative_operations(subgoal, operations)
    assert 2 <= len(selected) <= 4
    assert any(role == "direct_anchor" for _, _, _, role in selected)
    assert any(role == "explicit_conflict" for _, _, _, role in selected)


def test_role_complementary_selection_preserves_anchor_and_required_support():
    keys = [
        "route_search", "route_closure", "notebook_grounding",
        "lodging_search", "party_capacity", "final_constraint_audit",
    ]
    operations = [{"operation_key": key} for key in keys]
    ranked = list(range(len(keys)))
    reserved = {
        str(value)
        for phase in PHASES
        for field in ("anchor", "required_complement")
        for value in phase.get(field, ())
    }
    transport = select_role_complementary(
        ranked, operations, PHASES[0], signature(), set(), reserved, 2
    )
    assert [operations[index]["operation_key"] for index in transport] == [
        "route_search", "route_closure",
    ]
    budget_ranked = [5, 2, 0, 1, 3, 4]
    budget_operations = operations + [{"operation_key": "full_party_budget"}]
    budget_ranked = [6] + budget_ranked
    budget = select_role_complementary(
        budget_ranked, budget_operations, PHASES[-1], signature(), set(), reserved, 2
    )
    assert [budget_operations[index]["operation_key"] for index in budget] == [
        "full_party_budget", "final_constraint_audit",
    ]


def test_role_complementary_prefers_visible_conditional_role():
    operations = [
        {"operation_key": "lodging_search"},
        {"operation_key": "minimum_nights"},
        {"operation_key": "room_type"},
        {"operation_key": "house_rule"},
    ]
    reserved = {
        str(value)
        for phase in PHASES
        for field in ("anchor", "required_complement")
        for value in phase.get(field, ())
    }
    chosen = select_role_complementary(
        [0, 1, 2, 3], operations, PHASES[1], signature(), set(), reserved, 2
    )
    # The scorer still chooses between active room/house constraints, but a
    # generic minimum-night operation cannot displace both visible roles.
    assert operations[chosen[1]]["operation_key"] == "room_type"


def test_conditional_compatibility_generalizes_across_unseen_values():
    operation = {
        "applicability_field": "cuisine",
        "applicability_value": ["French"],
    }
    assert compatible(signature(cuisine=["Chinese", "Mexican"]), operation)
    assert not compatible(signature(cuisine=None), operation)
