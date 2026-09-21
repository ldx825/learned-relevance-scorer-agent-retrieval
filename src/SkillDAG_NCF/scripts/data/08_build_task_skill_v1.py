#!/usr/bin/env python3
"""Build task_skill_v1 locally; this script performs no API calls."""

from __future__ import annotations

import json
from pathlib import Path

from skilldag.data_pipeline import build_task_skill_dataset


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    data = root / ".runtime" / "skilldag_ncf" / "data"
    graph = root / ".runtime" / "skilldag" / "data" / "skilldag" / "skilldag_graphs" / "skillgraph_alfworld.json"
    report = build_task_skill_dataset(
        data / "raw" / "tasks.jsonl",
        data / "raw" / "skills.jsonl",
        data / "intermediate" / "train_combined_candidates.jsonl",
        data / "labels" / "medium_v1_judge_evidence.jsonl",
        graph,
        data / "reports" / "task_semantic_pilot_selection.json",
        data,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
