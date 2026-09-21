#!/usr/bin/env python3
"""Evaluate no-training task_skill_v1 baselines."""

from __future__ import annotations

import json
from pathlib import Path

from skilldag.data_pipeline.model_baselines import evaluate_model_baselines


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    data = root / ".runtime" / "skilldag_ncf" / "data"
    report = evaluate_model_baselines(
        data / "datasets" / "task_skill_v1",
        data / "model_data" / "task_skill_v1" / "arrays.npz",
        data,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
