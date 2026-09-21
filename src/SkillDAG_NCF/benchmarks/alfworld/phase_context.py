"""Deterministic ALFWorld full-task/phase context construction.

The functions in this module intentionally use only task type and PDDL goal
parameters. They never inspect expert plans, retrieval traces, rewards, or
evaluation outcomes. Training-data generation and the future online reranker
must share :func:`format_phase_context` to avoid train/runtime string drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


SUPPORTED_TASK_TYPES = {
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_and_place_with_movable_recep",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
}

STATE_BY_TASK_TYPE = {
    "pick_clean_then_place_in_recep": "clean",
    "pick_cool_then_place_in_recep": "cool",
    "pick_heat_then_place_in_recep": "heat",
}


def normalize_entity(value: Any) -> str:
    """Normalize ALFWorld CamelCase identifiers to environment vocabulary."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


@dataclass(frozen=True)
class TaskStructure:
    task_type: str
    object_name: str
    result_object_name: str
    object_count: int
    object_sliced: bool
    required_state: str
    target_receptacle: str
    movable_receptacle: str
    device: str
    canonical_full_task: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseSpec:
    phase_name: str
    phase_group: str
    phase_query: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _require(value: str, field: str, task_type: str) -> str:
    if not value:
        raise ValueError(f"missing {field} for task type {task_type}")
    return value


def build_task_structure(
    task_type: str,
    pddl_params: dict[str, Any],
) -> TaskStructure:
    """Build the retrieval-facing task structure without using an expert plan."""
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"unsupported ALFWorld task type: {task_type}")

    object_name = _require(
        normalize_entity(pddl_params.get("object_target")),
        "object_target",
        task_type,
    )
    object_sliced = bool(pddl_params.get("object_sliced", False))
    result_object_name = f"{object_name}sliced" if object_sliced else object_name
    object_count = 2 if task_type == "pick_two_obj_and_place" else 1
    required_state = STATE_BY_TASK_TYPE.get(task_type, "none")
    target_receptacle = normalize_entity(pddl_params.get("parent_target"))
    movable_receptacle = normalize_entity(pddl_params.get("mrecep_target"))
    device = normalize_entity(pddl_params.get("toggle_target"))

    sliced_text = f"sliced {object_name}" if object_sliced else object_name
    if task_type == "look_at_obj_in_light":
        device = _require(device, "toggle_target", task_type)
        canonical = f"examine {object_name} with {device}"
    elif task_type == "pick_and_place_with_movable_recep":
        movable_receptacle = _require(
            movable_receptacle, "mrecep_target", task_type
        )
        target_receptacle = _require(
            target_receptacle, "parent_target", task_type
        )
        canonical = (
            f"put {sliced_text} in {movable_receptacle}, then put "
            f"{movable_receptacle} in {target_receptacle}"
        )
    elif task_type == "pick_two_obj_and_place":
        target_receptacle = _require(
            target_receptacle, "parent_target", task_type
        )
        canonical = (
            f"put two distinct instances of {sliced_text} in "
            f"{target_receptacle}"
        )
    elif task_type in STATE_BY_TASK_TYPE:
        target_receptacle = _require(
            target_receptacle, "parent_target", task_type
        )
        canonical = (
            f"{required_state} {sliced_text} and put it in "
            f"{target_receptacle}"
        )
    else:
        target_receptacle = _require(
            target_receptacle, "parent_target", task_type
        )
        canonical = f"put {sliced_text} in {target_receptacle}"

    return TaskStructure(
        task_type=task_type,
        object_name=object_name,
        result_object_name=result_object_name,
        object_count=object_count,
        object_sliced=object_sliced,
        required_state=required_state,
        target_receptacle=target_receptacle,
        movable_receptacle=movable_receptacle,
        device=device,
        canonical_full_task=canonical,
    )


def _slice_phase(structure: TaskStructure) -> PhaseSpec:
    if structure.object_count == 2:
        query = f"slice two distinct {structure.object_name} objects with knife"
    else:
        query = f"slice {structure.object_name} with knife"
    return PhaseSpec("slice_object", "transform", query)


def build_phase_specs(structure: TaskStructure) -> list[PhaseSpec]:
    """Return ordered macro retrieval phases for one structured task."""
    task_type = structure.task_type
    obj = structure.object_name
    result_obj = structure.result_object_name
    target = structure.target_receptacle

    if task_type == "look_at_obj_in_light":
        return [
            PhaseSpec("locate_object", "locate", f"find {obj}"),
            PhaseSpec("locate_device", "locate", f"find {structure.device}"),
            PhaseSpec(
                "operate_device",
                "transform",
                f"examine {obj} with {structure.device}",
            ),
            PhaseSpec(
                "verify_goal",
                "verify",
                f"verify {obj} was examined with {structure.device}",
            ),
        ]

    if task_type == "pick_and_place_with_movable_recep":
        phases = [PhaseSpec("locate_object", "locate", f"find {obj}")]
        if structure.object_sliced:
            phases.append(_slice_phase(structure))
        phases.extend(
            [
                PhaseSpec(
                    "locate_movable_receptacle",
                    "locate",
                    f"find {structure.movable_receptacle}",
                ),
                PhaseSpec(
                    "assemble",
                    "transform",
                    f"put {result_obj} in {structure.movable_receptacle}",
                ),
                PhaseSpec(
                    "place_movable_receptacle",
                    "place",
                    f"put {structure.movable_receptacle} in {target}",
                ),
                PhaseSpec(
                    "verify_goal",
                    "verify",
                    f"verify {result_obj} is in {structure.movable_receptacle} "
                    f"and {structure.movable_receptacle} is in {target}",
                ),
            ]
        )
        return phases

    if task_type == "pick_two_obj_and_place":
        phases = [
            PhaseSpec(
                "locate_distinct_objects",
                "locate",
                f"find two distinct {obj} objects",
            )
        ]
        if structure.object_sliced:
            phases.append(_slice_phase(structure))
        phases.extend(
            [
                PhaseSpec(
                    "place_objects",
                    "place",
                    f"put both {result_obj} objects in {target}",
                ),
                PhaseSpec(
                    "verify_count",
                    "verify",
                    f"verify two distinct {result_obj} objects are in {target}",
                ),
            ]
        )
        return phases

    phases = [PhaseSpec("locate_object", "locate", f"find {obj}")]
    if structure.object_sliced:
        phases.append(_slice_phase(structure))

    if task_type in STATE_BY_TASK_TYPE:
        phases.append(
            PhaseSpec(
                "transform_object",
                "transform",
                f"{structure.required_state} {result_obj}",
            )
        )
    phases.extend(
        [
            PhaseSpec("place_object", "place", f"put {result_obj} in {target}"),
            PhaseSpec(
                "verify_goal",
                "verify",
                (
                    f"verify {result_obj} is {structure.required_state} and in {target}"
                    if structure.required_state != "none"
                    else f"verify {result_obj} is in {target}"
                ),
            ),
        ]
    )
    return phases


def format_phase_context(
    *,
    raw_full_task: str,
    structure: TaskStructure,
    phase: PhaseSpec,
) -> str:
    """Canonical text embedded by both V3 training and future online inference."""
    fields = [
        ("Full task", raw_full_task.strip()),
        ("Canonical goal", structure.canonical_full_task),
        ("Task type", structure.task_type),
        ("Current phase", phase.phase_name),
        ("Phase group", phase.phase_group),
        ("Current query", phase.phase_query),
        ("Object", structure.object_name),
        ("Object count", str(structure.object_count)),
        ("Object sliced", "yes" if structure.object_sliced else "no"),
        ("Required state", structure.required_state),
        ("Target receptacle", structure.target_receptacle or "none"),
        ("Movable receptacle", structure.movable_receptacle or "none"),
        ("Device", structure.device or "none"),
    ]
    return "\n".join(f"{name}: {value}" for name, value in fields)
