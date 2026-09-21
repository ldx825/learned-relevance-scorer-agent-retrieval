#!/usr/bin/env python3
"""Prepare leakage-aware model arrays from local caches; no API calls."""

from __future__ import annotations

import json
from pathlib import Path

from skilldag.data_pipeline.model_data import prepare_model_data


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    data = root / ".runtime" / "skilldag_ncf" / "data"
    graphs = root / ".runtime" / "skilldag" / "data" / "skilldag" / "skilldag_graphs"
    report = prepare_model_data(
        data / "datasets" / "task_skill_v1",
        data / "cache" / "task_embeddings" / "train",
        graphs / "skillgraph_alfworld.embeddings.json",
        graphs / "skillgraph_alfworld.json",
        data,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
