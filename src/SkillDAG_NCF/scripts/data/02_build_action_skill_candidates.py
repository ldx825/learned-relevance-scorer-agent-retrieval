#!/usr/bin/env python3
"""Stage 2: reuse skill vectors and embed unique ALFWorld plan actions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from skilldag.data_pipeline import build_action_skill_candidates
from skilldag.initialize import _embed_batch


def _defaults() -> tuple[Path, Path, Path, Path]:
    experiment_root = Path(__file__).resolve().parents[4]
    graph_root = experiment_root / ".runtime" / "skilldag" / "data" / "skilldag" / "skilldag_graphs"
    output_root = experiment_root / ".runtime" / "skilldag_ncf" / "data"
    return (
        output_root / "raw" / "tasks.jsonl",
        graph_root / "skillgraph_alfworld.json",
        graph_root / "skillgraph_alfworld.embeddings.json",
        output_root,
    )


def main() -> int:
    tasks_default, graph_default, embeddings_default, output_default = _defaults()
    parser = argparse.ArgumentParser(
        description="Build candidate-only action/skill tables; existing skill vectors are never recomputed."
    )
    parser.add_argument("--tasks-path", type=Path, default=tasks_default)
    parser.add_argument("--graph-path", type=Path, default=graph_default)
    parser.add_argument("--skill-embeddings-path", type=Path, default=embeddings_default)
    parser.add_argument("--output-root", type=Path, default=output_default)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--graph-seed-k",
        type=int,
        default=0,
        help="Expand one-hop neighbors from only this many highest-ranked semantic matches.",
    )
    args = parser.parse_args()

    model = os.environ.get("SKILLDAG_EMBEDDING_MODEL", "text-embedding-3-large")
    if not os.environ.get("SKILLDAG_EMBEDDING_API_KEY"):
        raise SystemExit(
            "SKILLDAG_EMBEDDING_API_KEY is empty. Run through "
            "scripts/data/02_build_action_skill_candidates.sh yunwu."
        )
    report = build_action_skill_candidates(
        tasks_path=args.tasks_path,
        graph_path=args.graph_path,
        skill_embeddings_path=args.skill_embeddings_path,
        output_root=args.output_root,
        embedding_model=model,
        embed=_embed_batch,
        top_k=args.top_k,
        graph_seed_k=args.graph_seed_k,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
