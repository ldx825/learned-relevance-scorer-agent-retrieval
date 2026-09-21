#!/usr/bin/env python3
"""Small CLI wrapper for the bundled ALFWorld SkillDAG runtime."""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ALFWorld with SkillDAG retrieval")
    parser.add_argument("--model", required=True)
    parser.add_argument("--skilldag_api_base", default="")
    parser.add_argument("--skills_dir", required=True)
    parser.add_argument("--skilldag_graph", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--max_games", type=int, default=140)
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--exp_name", default="skilldag")
    parser.add_argument("--task_indices", nargs="*", type=int)
    parser.add_argument("--skilldag_max_turns", type=int, default=0)
    parser.add_argument("--skilldag_auto_bootstrap", action="store_true")
    parser.add_argument("--skilldag_force_bootstrap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from benchmarks.alfworld.skilldag_runtime import run_skilldag

    run_skilldag(args)


if __name__ == "__main__":
    main()
