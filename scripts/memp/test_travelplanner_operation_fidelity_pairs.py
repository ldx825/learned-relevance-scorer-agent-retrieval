#!/usr/bin/env python3
"""Regression checks for V7 contextual operation fidelity scoring."""

from __future__ import annotations

import json
from pathlib import Path

from calibrate_travelplanner_operation_fidelity_pairs import fidelity_score


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "data/memp_ncf/travelplanner_atomic_memory_candidates_v4/memory_operations.jsonl"
operations = {row["operation_id"]: row for row in map(json.loads, OPS.read_text().splitlines())}

plain = {"signature": {"room_type": None, "house_rule": None, "transportation": None, "cuisine": None}}
private = {"signature": {"room_type": "private room", "house_rule": None, "transportation": None, "cuisine": None}}

assert fidelity_score(plain, operations["memory_001:route_search:00"]) > fidelity_score(
    plain, operations["memory_026:route_search:00"]
)
assert fidelity_score(plain, operations["memory_000:lodging_search:00"]) > fidelity_score(
    plain, operations["memory_018:lodging_search:00"]
)
assert fidelity_score(private, operations["memory_018:lodging_search:00"]) > fidelity_score(
    plain, operations["memory_018:lodging_search:00"]
)
assert fidelity_score(plain, operations["memory_024:notebook_grounding:00"]) > fidelity_score(
    plain, operations["memory_001:notebook_grounding:02"]
)
assert fidelity_score(plain, operations["memory_003:full_party_budget:00"]) > fidelity_score(
    plain, operations["memory_021:full_party_budget:00"]
)
print("TravelPlanner V7 operation-fidelity checks: PASS")
