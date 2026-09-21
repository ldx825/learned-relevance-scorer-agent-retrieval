#!/usr/bin/env python3
"""Generate deterministic, unlabeled phase records for NCF V3.

No API calls are made. No expert plan, online trajectory, reward, skill label,
or valid_seen/valid_unseen task text is written to the phase dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SKILLDAG_PROJECT = ROOT / "src/SkillDAG_NCF"
if str(SKILLDAG_PROJECT) not in sys.path:
    sys.path.insert(0, str(SKILLDAG_PROJECT))

from benchmarks.alfworld.phase_context import (  # noqa: E402
    build_phase_specs,
    build_task_structure,
    format_phase_context,
)


DATA_ROOT = ROOT / "data/alfworld_task_skill"
RAW_TASKS = DATA_ROOT / "shared/raw/tasks_all_official_splits.jsonl"
DEFAULT_MANIFEST_DIR = DATA_ROOT / "task_skill_v3_phase/manifests"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "task_skill_v3_phase/phases"
INTERNAL_SPLITS = ("train", "dev", "test")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest_path = args.manifest_dir / "source_manifest.json"
    forbidden_path = args.manifest_dir / "forbidden_hashes.json"
    for path in (RAW_TASKS, source_manifest_path, forbidden_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = load_json(source_manifest_path)
    forbidden = load_json(forbidden_path)
    tasks = load_jsonl(RAW_TASKS)
    task_by_id = {row["record_id"]: row for row in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("raw task archive contains duplicate record IDs")

    forbidden_text_hashes = {
        value
        for payload in forbidden["splits"].values()
        for value in payload["task_text_hashes"]
    }
    phase_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []

    for internal_split in INTERNAL_SPLITS:
        for record_id in source_manifest["record_ids"][internal_split]:
            source = task_by_id[record_id]
            if source["split"] != "train":
                raise ValueError(f"non-train task reached phase builder: {record_id}")
            if source["task_text_hash"] in forbidden_text_hashes:
                raise ValueError(f"forbidden task text reached phase builder: {record_id}")

            structure = build_task_structure(
                source["task_type"], source["pddl_params"]
            )
            phases = build_phase_specs(structure)
            phase_ids: list[str] = []
            for phase_index, phase in enumerate(phases):
                phase_id = (
                    f"{record_id}::phase-{phase_index:02d}-{phase.phase_name}"
                )
                phase_ids.append(phase_id)
                phase_rows.append(
                    {
                        "schema_version": "skilldag_ncf.v3.phase.v1",
                        "phase_id": phase_id,
                        "task_record_id": record_id,
                        "task_text_hash": source["task_text_hash"],
                        "official_split": "train",
                        "internal_split": internal_split,
                        "task_type": source["task_type"],
                        "phase_index": phase_index,
                        "phase_name": phase.phase_name,
                        "phase_group": phase.phase_group,
                        "phase_query": phase.phase_query,
                        "raw_full_task": source["task_text"],
                        "canonical_full_task": structure.canonical_full_task,
                        "context_text": format_phase_context(
                            raw_full_task=source["task_text"],
                            structure=structure,
                            phase=phase,
                        ),
                        "task_structure": structure.to_dict(),
                    }
                )

            task_rows.append(
                {
                    "schema_version": "skilldag_ncf.v3.phase_task.v1",
                    "task_record_id": record_id,
                    "task_text_hash": source["task_text_hash"],
                    "official_split": "train",
                    "internal_split": internal_split,
                    "task_type": source["task_type"],
                    "raw_full_task": source["task_text"],
                    "canonical_full_task": structure.canonical_full_task,
                    "task_structure": structure.to_dict(),
                    "phase_count": len(phases),
                    "phase_ids": phase_ids,
                }
            )

    task_rows.sort(key=lambda row: (row["internal_split"], row["task_record_id"]))
    phase_rows.sort(
        key=lambda row: (
            row["internal_split"],
            row["task_record_id"],
            row["phase_index"],
        )
    )
    phase_counts = Counter(row["phase_name"] for row in phase_rows)
    group_counts = Counter(row["phase_group"] for row in phase_rows)
    type_counts = Counter(row["task_type"] for row in task_rows)
    split_task_counts = Counter(row["internal_split"] for row in task_rows)
    split_phase_counts = Counter(row["internal_split"] for row in phase_rows)
    sliced_tasks = sum(row["task_structure"]["object_sliced"] for row in task_rows)

    report = {
        "schema_version": "skilldag_ncf.v3.phase_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": {
            "path": str(source_manifest_path.resolve().relative_to(ROOT)),
            "sha256": sha256_file(source_manifest_path),
        },
        "task_count": len(task_rows),
        "phase_count": len(phase_rows),
        "sliced_task_count": sliced_tasks,
        "task_counts_by_internal_split": dict(sorted(split_task_counts.items())),
        "phase_counts_by_internal_split": dict(sorted(split_phase_counts.items())),
        "task_counts_by_type": dict(sorted(type_counts.items())),
        "phase_counts_by_name": dict(sorted(phase_counts.items())),
        "phase_counts_by_group": dict(sorted(group_counts.items())),
        "contains_labels": False,
        "contains_skill_ids": False,
        "uses_expert_plan": False,
        "uses_online_trajectory": False,
        "api_calls_made": 0,
        "outputs": {
            "tasks": "tasks.jsonl",
            "phases": "phases.jsonl",
            "report": "report.json",
        },
    }
    dump_jsonl(args.output_dir / "tasks.jsonl", task_rows)
    dump_jsonl(args.output_dir / "phases.jsonl", phase_rows)
    dump_json(args.output_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build(args)
    print("V3 deterministic phase data: BUILT")
    print(f"output: {args.output_dir}")
    print(f"tasks: {report['task_count']}")
    print(f"phases: {report['phase_count']}")
    print(f"sliced tasks with extra phase: {report['sliced_task_count']}")
    print(f"phase groups: {report['phase_counts_by_group']}")
    print("labels created: 0")
    print("API calls made: 0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"V3 deterministic phase data: FAILED: {exc}")
        raise SystemExit(1)
