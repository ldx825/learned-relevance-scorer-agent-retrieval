#!/usr/bin/env python3
"""Stage 3 pilot: task-text top-K plus action top-K on 70 train tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from skilldag.data_pipeline import build_task_semantic_pilot
from skilldag.initialize import _embed_batch


def _defaults() -> tuple[Path, Path, Path, Path, Path]:
    experiment_root = Path(__file__).resolve().parents[4]
    graph_root = experiment_root / ".runtime" / "skilldag" / "data" / "skilldag" / "skilldag_graphs"
    output_root = experiment_root / ".runtime" / "skilldag_ncf" / "data"
    return (
        output_root / "raw" / "tasks.jsonl",
        output_root / "intermediate" / "task_skill_candidates.jsonl",
        graph_root / "skillgraph_alfworld.json",
        graph_root / "skillgraph_alfworld.embeddings.json",
        output_root,
    )


def main() -> int:
    tasks, action_candidates, graph, skill_embeddings, output = _defaults()
    parser = argparse.ArgumentParser(description="Run a stratified task-semantic retrieval pilot.")
    parser.add_argument("--tasks-path", type=Path, default=tasks)
    parser.add_argument("--action-candidates-path", type=Path, default=action_candidates)
    parser.add_argument("--graph-path", type=Path, default=graph)
    parser.add_argument("--skill-embeddings-path", type=Path, default=skill_embeddings)
    parser.add_argument("--output-root", type=Path, default=output)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-task-type", type=int, default=10)
    parser.add_argument("--task-top-k", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("SKILLDAG_EMBEDDING_API_KEY"):
        raise SystemExit("SKILLDAG_EMBEDDING_API_KEY is empty; run through the companion .sh script.")
    model = os.environ.get("SKILLDAG_EMBEDDING_MODEL", "text-embedding-3-large")
    report = build_task_semantic_pilot(
        args.tasks_path,
        args.action_candidates_path,
        args.graph_path,
        args.skill_embeddings_path,
        args.output_root,
        model,
        _embed_batch,
        split=args.split,
        samples_per_task_type=args.samples_per_task_type,
        task_top_k=args.task_top_k,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
