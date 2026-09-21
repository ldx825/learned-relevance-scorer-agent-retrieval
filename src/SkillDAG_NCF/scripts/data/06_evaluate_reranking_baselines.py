#!/usr/bin/env python3
"""Evaluate non-trained reranking baselines on Judge pilot evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skilldag.data_pipeline import evaluate_reranking_baselines


def _defaults() -> tuple[Path, Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[4]
    data = root / ".runtime" / "skilldag_ncf" / "data"
    graph = root / ".runtime" / "skilldag" / "data" / "skilldag" / "skilldag_graphs" / "skillgraph_alfworld.json"
    return (
        data / "raw" / "tasks.jsonl",
        data / "intermediate" / "pilot_combined_candidates.jsonl",
        data / "labels" / "pilot_judge_evidence.jsonl",
        graph,
        data,
    )


def main() -> int:
    tasks, candidates, evidence, graph, output = _defaults()
    parser = argparse.ArgumentParser(description="Run offline candidate reranking baselines.")
    parser.add_argument("--tasks-path", type=Path, default=tasks)
    parser.add_argument("--candidates-path", type=Path, default=candidates)
    parser.add_argument("--evidence-path", type=Path, default=evidence)
    parser.add_argument("--graph-path", type=Path, default=graph)
    parser.add_argument("--output-root", type=Path, default=output)
    args = parser.parse_args()
    report = evaluate_reranking_baselines(
        args.tasks_path,
        args.candidates_path,
        args.evidence_path,
        args.graph_path,
        args.output_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
