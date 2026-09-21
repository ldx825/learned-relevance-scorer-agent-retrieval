#!/usr/bin/env python3
"""Prepare the deterministic 100-per-task-type Judge selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skilldag.data_pipeline import build_stratified_selection


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    data = root / ".runtime" / "skilldag_ncf" / "data"
    parser = argparse.ArgumentParser(description="Build a stratified Judge task selection.")
    parser.add_argument("--tasks-path", type=Path, default=data / "raw" / "tasks.jsonl")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=data / "reports" / "judge_medium_v1_selection.json",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-task-type", type=int, default=100)
    args = parser.parse_args()
    report = build_stratified_selection(
        args.tasks_path,
        args.output_path,
        split=args.split,
        samples_per_task_type=args.samples_per_task_type,
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "record_ids"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
