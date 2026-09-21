#!/usr/bin/env python3
"""Prepare generic task-type/phase templates for economical V3 judging."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/alfworld_task_skill"
DEFAULT_PHASES = DATA_ROOT / "task_skill_v3_phase/phases/phases.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "task_skill_v3_phase/judge/templates"


WORKFLOWS = {
    "look_at_obj_in_light": "locate a target object and a light device, then examine the object under the activated light",
    "pick_and_place_simple": "locate and acquire a target object, optionally slice it, then place it in a target receptacle",
    "pick_and_place_with_movable_recep": "locate an object and a movable container, optionally slice the object, put it in the movable container, then place that container in the final receptacle",
    "pick_clean_then_place_in_recep": "locate an object, optionally slice it, clean it with the appropriate appliance, then place it in the target receptacle",
    "pick_cool_then_place_in_recep": "locate an object, optionally slice it, cool it with the appropriate appliance, then place it in the target receptacle",
    "pick_heat_then_place_in_recep": "locate an object, optionally slice it, heat it with the appropriate appliance, then place it in the target receptacle",
    "pick_two_obj_and_place": "locate and acquire two distinct instances of an object, optionally slice them, then place both in the target receptacle",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def phase_objective(row: dict[str, Any]) -> str:
    name = row["phase_name"]
    state = row["task_structure"]["required_state"]
    objectives = {
        "locate_object": "find the target object",
        "locate_distinct_objects": "find two distinct instances of the target object",
        "locate_device": "find the required light device",
        "locate_movable_receptacle": "find the required movable container",
        "slice_object": "obtain and use the appropriate tool to slice the target object",
        "assemble": "put the target object inside the movable container",
        "place_object": "put the target object in the final target receptacle",
        "place_objects": "put both distinct target objects in the final target receptacle",
        "place_movable_receptacle": "put the movable container in the final target receptacle",
        "operate_device": "activate/use the light device and examine the target object",
        "verify_goal": "verify that the complete task goal and required state are satisfied",
        "verify_count": "verify that two distinct target objects are in the target receptacle",
    }
    if name == "transform_object":
        return f"{state} the target object with the appropriate appliance"
    return objectives[name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases-path", type=Path, default=DEFAULT_PHASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    phases = load_jsonl(args.phases_path)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in phases:
        grouped[(row["task_type"], row["phase_name"])].append(row)

    templates = []
    covered = 0
    for (task_type, phase_name), rows in sorted(grouped.items()):
        representative = rows[0]
        template_id = f"{task_type}::{phase_name}"
        templates.append(
            {
                "schema_version": "skilldag_ncf.v3.phase_judge_template.v1",
                "template_id": template_id,
                "task_type": task_type,
                "phase_name": phase_name,
                "phase_group": representative["phase_group"],
                "generic_full_workflow": WORKFLOWS[task_type],
                "generic_phase_objective": phase_objective(representative),
                "covered_phase_count": len(rows),
                "covered_phase_ids": sorted(row["phase_id"] for row in rows),
                "uses_specific_task_entity": False,
                "uses_expert_plan": False,
                "uses_eval_data": False,
            }
        )
        covered += len(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    templates_path = args.output_dir / "templates.jsonl"
    templates_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in templates),
        encoding="utf-8",
    )
    counts = Counter(row["phase_group"] for row in templates)
    report = {
        "schema_version": "skilldag_ncf.v3.phase_judge_templates_report.v1",
        "template_count": len(templates),
        "covered_phase_count": covered,
        "templates_by_phase_group": dict(sorted(counts.items())),
        "potential_judgments_at_37_skills_each": len(templates) * 37,
        "api_calls_made": 0,
        "uses_specific_task_entities": False,
        "uses_expert_plan": False,
        "uses_eval_data": False,
        "outputs": {"templates": "templates.jsonl", "report": "report.json"},
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
