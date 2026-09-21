#!/usr/bin/env python3
"""Build task/action candidates for every ALFWorld train task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from skilldag.data_pipeline import build_full_split_task_candidates
from skilldag.initialize import _embed_batch


def _defaults() -> tuple[Path, Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[4]
    data = root / ".runtime" / "skilldag_ncf" / "data"
    graphs = root / ".runtime" / "skilldag" / "data" / "skilldag" / "skilldag_graphs"
    return (
        data / "raw" / "tasks.jsonl",
        data / "intermediate" / "task_skill_candidates.jsonl",
        graphs / "skillgraph_alfworld.json",
        graphs / "skillgraph_alfworld.embeddings.json",
        data,
    )


def main() -> int:
    tasks, actions, graph, skills, output = _defaults()
    parser = argparse.ArgumentParser(description="Build resumable full-split task candidates.")
    parser.add_argument("--tasks-path", type=Path, default=tasks)
    parser.add_argument("--action-candidates-path", type=Path, default=actions)
    parser.add_argument("--graph-path", type=Path, default=graph)
    parser.add_argument("--skill-embeddings-path", type=Path, default=skills)
    parser.add_argument("--output-root", type=Path, default=output)
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if not os.environ.get("SKILLDAG_EMBEDDING_API_KEY"):
        raise SystemExit("SKILLDAG_EMBEDDING_API_KEY is empty; use the companion .sh script.")

    def show_progress(done: int, total: int, batch_tasks: int, api_inputs: int) -> None:
        print(
            f"[task-embedding] batch {done}/{total}: tasks={batch_tasks}, "
            f"new_api_inputs={api_inputs}",
            flush=True,
        )

    report = build_full_split_task_candidates(
        args.tasks_path,
        args.action_candidates_path,
        args.graph_path,
        args.skill_embeddings_path,
        args.output_root,
        os.environ.get("SKILLDAG_EMBEDDING_MODEL", "text-embedding-3-large"),
        _embed_batch,
        split=args.split,
        task_top_k=args.task_top_k,
        batch_size=args.batch_size,
        progress=show_progress,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
