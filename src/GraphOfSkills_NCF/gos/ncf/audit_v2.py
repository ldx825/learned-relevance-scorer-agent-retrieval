"""Optional ALFWorld semantic checks for V2 Judge labels.

This audit rejects suspicious labels; it does not create or rewrite labels.
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from .data_v2 import atomic_json, load_jsonl

TRANSFORM_PATTERNS = {
    "heat": (
        r"\b(heat|heated|cook|cooked|warm|warmed|microwaved|"
        r"toast|toasted)\b"
        r"|(?:^|\b(?:then|and|to)\s+)microwave\b"
    ),
    "cool": (
        r"\b(cool|cooled|chill|chilled|cold|refrigerate|refrigerated)\b"
        r"|\bin (?:the )?fridge for (?:a )?(?:bit|while)\b"
    ),
    "clean": r"\b(clean|cleaned|wash|washed|rinse|rinsed|wet)\b",
}
TRANSFORM_SKILLS = {
    "heat": {
        "alfworld-object-heater",
        "alfworld-heat-object-with-appliance",
        "alfworld-temperature-regulator",
    },
    "cool": {"alfworld-object-cooler", "alfworld-temperature-regulator"},
    "clean": {"alfworld-clean-object"},
}
TASK_TYPE_TRANSFORMS = {
    "pick_heat_then_place_in_recep": {"heat"},
    "pick_cool_then_place_in_recep": {"cool"},
    "pick_clean_then_place_in_recep": {"clean"},
}


def requested_transformations(task: dict[str, Any]) -> set[str]:
    """Infer required capabilities from public task text plus ALFWorld goal type.

    The task type is used as a consistency signal for terse annotations such
    as ``Move pot to refrigerator then to counter``.  Slicing is intentionally
    not audited because the current 37-skill library has no slicing skill;
    mapping it to the generic ``tool-user`` skill produced false violations.
    """
    text = str(task.get("task_text", "")).lower()
    requested = {
        name for name, pattern in TRANSFORM_PATTERNS.items() if re.search(pattern, text)
    }
    requested.update(TASK_TYPE_TRANSFORMS.get(str(task.get("task_type", "")), set()))
    return requested


def audit_gos_v2_labels(
    tasks_path: Path | str,
    labels_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    tasks = {row["record_id"]: row for row in load_jsonl(Path(tasks_path))}
    grouped = defaultdict(list)
    for row in load_jsonl(Path(labels_path)):
        grouped[row["task_record_id"]].append(row)
    violations = []
    coverage = Counter()
    primary_frequency = Counter()
    for task_id, rows in grouped.items():
        requested = requested_transformations(tasks[task_id])
        positive = {
            row["skill_id"] for row in rows if row["role"] in {"primary", "support"}
        }
        primary = {row["skill_id"] for row in rows if row["role"] == "primary"}
        primary_frequency.update(primary)
        requested_transform_skills = set().union(
            *(TRANSFORM_SKILLS[operation] for operation in requested)
        ) if requested else set()
        for operation in requested:
            coverage["expected"] += 1
            if primary & TRANSFORM_SKILLS[operation]:
                coverage["primary_hit"] += 1
            else:
                violations.append({
                    "task_record_id": task_id,
                    "task_text": tasks[task_id]["task_text"],
                    "type": "requested_transformation_missing_from_primary",
                    "operation": operation,
                })
        for operation, skill_ids in TRANSFORM_SKILLS.items():
            unexpected_positive = (positive & skill_ids) - requested_transform_skills
            if operation not in requested and unexpected_positive:
                violations.append({
                    "task_record_id": task_id,
                    "task_text": tasks[task_id]["task_text"],
                    "type": "unrequested_transformation_marked_positive",
                    "operation": operation,
                    "skill_ids": sorted(unexpected_positive),
                })
    report = {
        "task_count": len(grouped),
        "pair_count": sum(len(rows) for rows in grouped.values()),
        "violation_count": len(violations),
        "violations": violations,
        "requested_transformation_primary_recall": (
            coverage["primary_hit"] / coverage["expected"] if coverage["expected"] else 1.0
        ),
        "primary_skill_frequency": dict(primary_frequency.most_common()),
        "max_primary_task_rate": (
            max(primary_frequency.values()) / len(grouped) if primary_frequency else 0.0
        ),
        "passed": not violations,
    }
    atomic_json(Path(output_path), report)
    return report
