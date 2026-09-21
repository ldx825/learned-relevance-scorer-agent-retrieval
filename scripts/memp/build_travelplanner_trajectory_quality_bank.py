#!/usr/bin/env python3
"""Build the V26 quality-gated trajectory bank (41 train trajectories).

Takes the best evaluation per train task from train_trajectory_full (across
attempts), renders the executed trajectory steps as retrieval content, and
attaches the reconstructed Memp final score as an execution-quality signal.

This is the train-side bank for the V26 label: min(quality, match).  No
official validation/test artifact is read.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_travelplanner_trajectory_content_texts import render_steps  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("trajectory", "proceduralization"),
        default="trajectory",
        help="content rendering style",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all-attempts",
        action="store_true",
        help="include every evaluated attempt per task (expanded pool)",
    )
    parser.add_argument(
        "--quality-grade-map",
        type=str,
        default="0.8:2,0.6:1,0.0:0",
        help="comma list of threshold:grade, highest first",
    )
    args = parser.parse_args()

    pairs = [
        (float(threshold), int(grade))
        for chunk in args.quality_grade_map.split(",")
        for threshold, grade in [chunk.split(":")]
    ]
    pairs.sort(key=lambda pair: -pair[0])

    def quality_grade(score: float) -> int:
        for threshold, grade in pairs:
            if score >= threshold:
                return grade
        return 0

    best: dict[int, tuple[str, float, str]] = {}
    best_all: dict[int, list[tuple[str, float, str]]] = {}
    for evaluation_path in glob.glob(
        str(args.trajectory_root / "attempt_*" / "*_evaluation.json")
    ):
        evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
        train_index = int(evaluation["train_index"])
        score = float(evaluation.get("reconstructed_memp_final_score") or 0.0)
        trajectory_path = evaluation_path[: -len("_evaluation.json")] + ".json"
        if not Path(trajectory_path).exists():
            continue
        best_all.setdefault(train_index, []).append((trajectory_path, score, evaluation_path))
        if train_index not in best or score > best[train_index][1]:
            best[train_index] = (trajectory_path, score, evaluation_path)

    rows: list[dict[str, Any]] = []
    if args.all_attempts:
        entries = []
        for train_index, lst in best_all.items():
            for trajectory_path, score, evaluation_path in lst:
                attempt = Path(evaluation_path).parent.name
                entries.append((train_index, attempt, trajectory_path, score))
        entries.sort(key=lambda e: (e[0], e[1]))
    else:
        entries = [(ti, "best", best[ti][0], best[ti][1]) for ti in sorted(best)]
    for train_index, attempt, trajectory_path, score in entries:
        trajectory = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
        steps = trajectory.get("trajectory")
        if not isinstance(steps, list) or not steps:
            continue
        if args.format == "proceduralization":
            plan_text = str(trajectory.get("final_plan_text") or "").strip()
            chunks = []
            if plan_text:
                chunks.append("Proceduralized plan:\n" + plan_text)
            chunks.append("Executed trajectory:\n" + render_steps(steps))
            content = "\n\n".join(chunks)
        else:
            content = "Executed trajectory:\n" + render_steps(steps)
        rows.append(
            {
                "memory_id": f"{args.format}_q_{train_index:03d}_{attempt}",
                "source": f"travelplanner_train_{train_index:03d}",
                "source_query": str(trajectory.get("query") or "").strip(),
                "content_text": content,
                "quality_score": round(score, 4),
                "quality_grade": quality_grade(score),
                "execution_quality_source": "train_reconstructed_memp_final_score",
                "validation_or_test_used": False,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    high = sum(row["quality_grade"] == 2 for row in rows)
    low = sum(row["quality_grade"] == 0 for row in rows)
    print(f"wrote {len(rows)} trajectory documents (grade2={high}, grade1={len(rows)-high-low}, grade0={low})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
