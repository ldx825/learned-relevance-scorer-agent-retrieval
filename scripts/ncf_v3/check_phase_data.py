#!/usr/bin/env python3
"""Audit deterministic V3 phase records without making API calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "alfworld_task_skill"
RAW_TASKS = DATA_ROOT / "shared/raw/tasks_all_official_splits.jsonl"
DEFAULT_MANIFEST_DIR = DATA_ROOT / "task_skill_v3_phase/manifests"
DEFAULT_PHASE_DIR = DATA_ROOT / "task_skill_v3_phase/phases"
INTERNAL_SPLITS = ("train", "dev", "test")

SKILLDAG_PROJECT = ROOT / "src/SkillDAG_NCF"
if str(SKILLDAG_PROJECT) not in sys.path:
    sys.path.insert(0, str(SKILLDAG_PROJECT))

from benchmarks.alfworld.phase_context import (  # noqa: E402
    build_phase_specs,
    build_task_structure,
    format_phase_context,
)

BASE_PHASE_COUNTS = {
    "look_at_obj_in_light": 4,
    "pick_and_place_simple": 3,
    "pick_and_place_with_movable_recep": 5,
    "pick_clean_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 4,
    "pick_heat_then_place_in_recep": 4,
    "pick_two_obj_and_place": 3,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = load_json(args.manifest_dir / "source_manifest.json")
    forbidden = load_json(args.manifest_dir / "forbidden_hashes.json")
    raw_tasks = load_jsonl(RAW_TASKS)
    raw_by_id = {row["record_id"]: row for row in raw_tasks}
    tasks = load_jsonl(args.phase_dir / "tasks.jsonl")
    phases = load_jsonl(args.phase_dir / "phases.jsonl")
    report = load_json(args.phase_dir / "report.json")

    source_ids = {
        split: set(source_manifest["record_ids"][split])
        for split in INTERNAL_SPLITS
    }
    output_ids: dict[str, set[str]] = {
        split: {
            row["task_record_id"]
            for row in tasks
            if row["internal_split"] == split
        }
        for split in INTERNAL_SPLITS
    }
    if source_ids != output_ids:
        raise ValueError("phase task IDs differ from the leakage-safe source manifest")
    if len(tasks) != 700 or len({row["task_record_id"] for row in tasks}) != 700:
        raise ValueError("phase task table must contain exactly 700 unique tasks")

    forbidden_hashes = {
        value
        for payload in forbidden["splits"].values()
        for value in payload["task_text_hashes"]
    }
    leaked = {row["task_text_hash"] for row in tasks} & forbidden_hashes
    if leaked:
        raise ValueError(f"phase dataset contains {len(leaked)} forbidden text hashes")

    phase_ids = [row["phase_id"] for row in phases]
    if len(phase_ids) != len(set(phase_ids)):
        raise ValueError("duplicate phase IDs")
    phases_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in phases:
        phases_by_task[row["task_record_id"]].append(row)
        if row["official_split"] != "train":
            raise ValueError("non-train phase record detected")
        for field in (
            "phase_name",
            "phase_group",
            "phase_query",
            "raw_full_task",
            "canonical_full_task",
            "context_text",
        ):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"empty {field} in {row['phase_id']}")
        forbidden_fields = {
            "target_grade",
            "label",
            "skill_id",
            "reward",
            "plan_high_pddl",
            "annotations",
            "trajectory",
        }
        present = forbidden_fields & set(row)
        if present:
            raise ValueError(f"forbidden fields in phase record: {sorted(present)}")

    for task in tasks:
        source = raw_by_id[task["task_record_id"]]
        expected_structure = build_task_structure(
            source["task_type"], source["pddl_params"]
        )
        if task["task_structure"] != expected_structure.to_dict():
            raise ValueError(
                f"persisted task structure drift: {task['task_record_id']}"
            )
        if task["raw_full_task"] != source["task_text"]:
            raise ValueError(f"raw task text drift: {task['task_record_id']}")
        if task["canonical_full_task"] != expected_structure.canonical_full_task:
            raise ValueError(f"canonical task drift: {task['task_record_id']}")

        task_phases = sorted(
            phases_by_task[task["task_record_id"]],
            key=lambda row: row["phase_index"],
        )
        expected_specs = build_phase_specs(expected_structure)
        expected_count = len(expected_specs)
        if len(task_phases) != expected_count:
            raise ValueError(
                f"unexpected phase count for {task['task_record_id']}: "
                f"expected={expected_count}, actual={len(task_phases)}"
            )
        if [row["phase_index"] for row in task_phases] != list(range(expected_count)):
            raise ValueError(f"non-contiguous phase indices: {task['task_record_id']}")
        if task["phase_ids"] != [row["phase_id"] for row in task_phases]:
            raise ValueError(f"task/phase ID mismatch: {task['task_record_id']}")
        for row, expected_spec in zip(task_phases, expected_specs, strict=True):
            expected_fields = expected_spec.to_dict()
            for field in ("phase_name", "phase_group", "phase_query"):
                if row[field] != expected_fields[field]:
                    raise ValueError(
                        f"persisted {field} drift: {row['phase_id']}"
                    )
            expected_context = format_phase_context(
                raw_full_task=source["task_text"],
                structure=expected_structure,
                phase=expected_spec,
            )
            if row["context_text"] != expected_context:
                raise ValueError(f"phase context drift: {row['phase_id']}")

        phase_names = [row["phase_name"] for row in task_phases]
        if phase_names[-1] not in {"verify_goal", "verify_count"}:
            raise ValueError(f"task does not end in verification: {task['task_record_id']}")
        if ("slice_object" in phase_names) != expected_structure.object_sliced:
            raise ValueError(f"slice phase/goal mismatch: {task['task_record_id']}")

    task_split_counts = Counter(row["internal_split"] for row in tasks)
    expected_split_counts = Counter({"train": 560, "dev": 70, "test": 70})
    if task_split_counts != expected_split_counts:
        raise ValueError(f"unexpected task split counts: {task_split_counts}")
    type_counts = Counter(row["task_type"] for row in tasks)
    if set(type_counts) != set(BASE_PHASE_COUNTS) or any(
        count != 100 for count in type_counts.values()
    ):
        raise ValueError(f"unexpected task-type counts: {type_counts}")
    if report["task_count"] != len(tasks) or report["phase_count"] != len(phases):
        raise ValueError("phase report counts disagree with data files")
    if any(
        report[key]
        for key in ("contains_labels", "contains_skill_ids", "uses_expert_plan", "uses_online_trajectory")
    ):
        raise ValueError("phase report claims forbidden information was used")

    return {
        "task_count": len(tasks),
        "phase_count": len(phases),
        "task_split_counts": dict(sorted(task_split_counts.items())),
        "task_type_counts": dict(sorted(type_counts.items())),
        "phase_group_counts": dict(
            sorted(Counter(row["phase_group"] for row in phases).items())
        ),
        "sliced_task_count": sum(
            row["task_structure"]["object_sliced"] for row in tasks
        ),
        "forbidden_text_overlap": 0,
        "api_calls_made": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--phase-dir", type=Path, default=DEFAULT_PHASE_DIR)
    return parser.parse_args()


def main() -> int:
    result = check(parse_args())
    print("[OK] V3 phase tasks match the safe source manifest")
    print(f"[OK] Tasks: {result['task_count']} {result['task_split_counts']}")
    print(f"[OK] Phases: {result['phase_count']} {result['phase_group_counts']}")
    print(f"[OK] Seven task types: {result['task_type_counts']}")
    print(f"[OK] Sliced tasks received an extra phase: {result['sliced_task_count']}")
    print(f"[OK] Official evaluation text overlap: {result['forbidden_text_overlap']}")
    print("[OK] All persisted task structures, phases, and contexts replay exactly")
    print("[OK] Labels=0, skill IDs=0, expert plans=0, online trajectories=0")
    print("[PASS] V3 deterministic phase data is valid")
    print("API calls made: 0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[FAILED] V3 phase audit: {exc}")
        raise SystemExit(1)
