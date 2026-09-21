#!/usr/bin/env python3
"""Stage 1: extract ALFWorld task and SkillDAG skill metadata locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skilldag.data_pipeline import extract_static_data


def _defaults() -> tuple[Path, Path, Path, Path]:
    experiment_root = Path(__file__).resolve().parents[4]
    skilldag_data = experiment_root / ".runtime" / "skilldag" / "data"
    return (
        skilldag_data / "alfworld" / "json_2.1.1",
        skilldag_data / "skilldag" / "alfworld_skills",
        skilldag_data / "skilldag" / "skilldag_graphs" / "skillgraph_alfworld.json",
        experiment_root / ".runtime" / "skilldag_ncf" / "data",
    )


def main() -> int:
    alfworld_default, skills_default, graph_default, output_default = _defaults()
    parser = argparse.ArgumentParser(
        description="Extract static task/skill JSONL files without any API calls."
    )
    parser.add_argument("--alfworld-root", type=Path, default=alfworld_default)
    parser.add_argument("--skills-dir", type=Path, default=skills_default)
    parser.add_argument("--graph-path", type=Path, default=graph_default)
    parser.add_argument("--output-root", type=Path, default=output_default)
    args = parser.parse_args()
    report = extract_static_data(
        alfworld_root=args.alfworld_root,
        skills_dir=args.skills_dir,
        graph_path=args.graph_path,
        output_root=args.output_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
