#!/usr/bin/env python3
"""Shared ALFWorld task-memory schema for MemP retrieval supervision.

The parser intentionally uses only the public ALFWorld training queries and
the frozen memory text.  It does not depend on dev/test rewards or trajectories.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskSignature:
    operation: str
    object_type: str
    cardinality: int
    destination: str

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)

    @property
    def canonical_id(self) -> str:
        return "|".join(
            (self.operation, self.object_type, str(self.cardinality), self.destination)
        )


PATTERNS: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    ("clean", 1, re.compile(r"^put a clean ([a-z0-9]+) (?:in|on) ([a-z0-9]+)$")),
    ("clean", 1, re.compile(r"^clean some ([a-z0-9]+) and put it (?:in|on) ([a-z0-9]+)$")),
    ("heat", 1, re.compile(r"^put a hot ([a-z0-9]+) (?:in|on) ([a-z0-9]+)$")),
    ("heat", 1, re.compile(r"^heat some ([a-z0-9]+) and put it (?:in|on) ([a-z0-9]+)$")),
    ("cool", 1, re.compile(r"^put a cool ([a-z0-9]+) (?:in|on) ([a-z0-9]+)$")),
    ("cool", 1, re.compile(r"^cool some ([a-z0-9]+) and put it (?:in|on) ([a-z0-9]+)$")),
    ("pick_two", 2, re.compile(r"^put two ([a-z0-9]+) (?:in|on) ([a-z0-9]+)$")),
    ("pick_two", 2, re.compile(r"^find two ([a-z0-9]+) and put them (?:in|on) ([a-z0-9]+)$")),
    ("examine", 1, re.compile(r"^look at ([a-z0-9]+) under the ([a-z0-9]+)$")),
    ("examine", 1, re.compile(r"^examine the ([a-z0-9]+) with the ([a-z0-9]+)$")),
    ("pick_place", 1, re.compile(r"^put an? ([a-z0-9]+) (?:in|on) ([a-z0-9]+)$")),
    ("pick_place", 1, re.compile(r"^put some ([a-z0-9]+) (?:in|on) ([a-z0-9]+)$")),
)


def normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().rstrip(".").split())


def parse_query(query: str) -> TaskSignature:
    normalized = normalize_query(query)
    for operation, cardinality, pattern in PATTERNS:
        match = pattern.fullmatch(normalized)
        if match:
            object_type, destination = match.groups()
            return TaskSignature(operation, object_type, cardinality, destination)
    raise ValueError(f"Unsupported ALFWorld query: {query!r}")


INVALID_SCRIPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "collect_all_before_placing",
        re.compile(
            r"(?:until all required objects are (?:collected|in your possession)|"
            r"once (?:all|both) (?:objects|items|cellphones) (?:are|have been) "
            r"(?:collected|retrieved|found)|all objects (?:are )?in hand)"
        ),
    ),
    (
        "simultaneous_multi_object_holding",
        re.compile(r"(?:hold|carry|pick up|take) (?:both|all) (?:objects|items)"),
    ),
)


def script_validity(signature: TaskSignature, workflow: str) -> tuple[bool, str]:
    """Conservative rule check for known ALFWorld-incompatible scripts."""
    if signature.operation != "pick_two":
        return True, "no_known_violation"
    text = normalize_query(workflow)
    for reason, pattern in INVALID_SCRIPT_PATTERNS:
        if pattern.search(text):
            return False, reason
    return True, "no_known_violation"


def relation_grade(
    task: TaskSignature,
    memory: TaskSignature,
    *,
    memory_is_valid: bool = True,
) -> tuple[int, str]:
    """Return a deterministic 2/1/0 task-memory supervision label.

    Grade 2 is an exact, executable match.  Grade 1 is useful complementary
    evidence.  Grade 0 is conflicting, invalid, redundant, or unrelated.
    """
    if task == memory:
        if memory_is_valid:
            return 2, "exact_executable_match"
        return 0, "exact_but_procedurally_invalid"

    same_operation = task.operation == memory.operation
    same_object = task.object_type == memory.object_type
    same_destination = task.destination == memory.destination

    if same_operation and same_object:
        return 1, "same_operation_object_different_destination"
    if same_operation and same_destination:
        return 1, "same_operation_destination_different_object"

    # A single-object placement can still supply object/destination navigation
    # evidence for a pick-two task, but it must never be treated as complete.
    if (
        task.operation == "pick_two"
        and memory.operation == "pick_place"
        and same_object
        and same_destination
    ):
        return 1, "single_object_subprocedure_for_pick_two"

    # Plain placement is a useful subprocedure after a transformation.
    if (
        task.operation in {"clean", "heat", "cool"}
        and memory.operation == "pick_place"
        and same_object
        and same_destination
    ):
        return 1, "placement_subprocedure_for_transformation"

    if same_object and same_destination:
        return 0, "operation_or_cardinality_conflict"
    if same_object:
        return 0, "same_object_wrong_relation"
    if same_destination:
        return 0, "same_destination_wrong_relation"
    if same_operation:
        return 0, "same_operation_unmatched_arguments"
    return 0, "unrelated"


def negative_hardness(task: TaskSignature, memory: TaskSignature) -> int:
    """Prioritize confusing grade-0 pairs over trivial random negatives."""
    score = 0
    score += 8 * int(task.object_type == memory.object_type)
    score += 6 * int(task.destination == memory.destination)
    score += 4 * int(task.operation == memory.operation)
    score += 2 * int(task.cardinality == memory.cardinality)
    return score
