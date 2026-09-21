#!/usr/bin/env python3
"""Small invariants for typed TravelPlanner hard-pair construction."""

from build_travelplanner_atomic_typed_hard_pairs import procedure_fingerprint, structural_distance


def test_structural_distance_identity() -> None:
    signature = {
        "days": 3,
        "visiting_city_number": 1,
        "people_number": 2,
        "budget_per_person_day": 100.0,
    }
    assert structural_distance(signature, signature) == 0.0


def test_structural_distance_orders_mismatch() -> None:
    query = {
        "days": 3,
        "visiting_city_number": 1,
        "people_number": 2,
        "budget_per_person_day": 100.0,
    }
    close = {**query, "days": 4}
    far = {
        "days": 7,
        "visiting_city_number": 3,
        "people_number": 8,
        "budget_per_person_day": 300.0,
    }
    assert structural_distance(query, close) < structural_distance(query, far)


def test_procedure_fingerprint_ignores_formatting_only() -> None:
    left = {"procedure": "Check every route-leg, then return."}
    right = {"procedure": " check EVERY route leg then return "}
    assert procedure_fingerprint(left) == procedure_fingerprint(right)
