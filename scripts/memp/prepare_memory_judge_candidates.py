#!/usr/bin/env python3
"""Convert train-only rule seeds into unlabeled Memory-Task Judge batches.

This mirrors the Skill-Task pipeline boundary: candidate generation and final
semantic labeling are separate stages.  Rule seed grades are retained only for
post-Judge calibration/audit and are not shown to the Judge.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def candidate_sources(pair: dict) -> list[str]:
    grade = int(pair["grade"])
    reason = pair["reason"]
    if grade == 2:
        return ["structured_exact"]
    if grade == 1:
        return ["structured_complement"]
    if reason == "unrelated":
        return ["deterministic_negative"]
    return ["structured_hard_negative"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tasks = {int(row["task_id"]): row for row in load_jsonl(args.dataset / "tasks.jsonl")}
    memories = {int(row["memory_id"]): row for row in load_jsonl(args.dataset / "memories.jsonl")}
    pairs_by_task: dict[int, list[dict]] = defaultdict(list)
    for pair in load_jsonl(args.dataset / "pairs.jsonl"):
        pairs_by_task[int(pair["task_id"])].append(pair)

    batches = []
    source_counts: Counter[str] = Counter()
    for task_id in sorted(tasks):
        task = tasks[task_id]
        candidates = []
        for pair in sorted(pairs_by_task[task_id], key=lambda row: int(row["memory_id"])):
            memory = memories[int(pair["memory_id"])]
            sources = candidate_sources(pair)
            source_counts.update(sources)
            candidates.append(
                {
                    "memory_id": memory["memory_id"],
                    "memory_query": memory["query"],
                    "workflow": memory["workflow"],
                    "candidate_sources": sources,
                    # Private calibration metadata: the Judge prompt builder
                    # deliberately omits every rule_seed_* field.
                    "rule_seed_grade": pair["grade"],
                    "rule_seed_reason": pair["reason"],
                    "procedurally_valid_rule": memory["procedurally_valid"],
                    "validity_reason_rule": memory["validity_reason"],
                }
            )
        batches.append(
            {
                "schema_version": "task_resource_ncf.memory_judge_candidate.v1",
                "task_id": task_id,
                "task_query": task["query"],
                "task_signature": task["signature"],
                "canonical_id": task["canonical_id"],
                "internal_split": task["model_split"],
                "source": task["source"],
                "candidates": candidates,
                "contains_final_labels": False,
                "uses_eval_data": False,
            }
        )

    write_jsonl(args.output_dir / "judge_candidates.jsonl", batches)
    report = {
        "schema_version": "task_resource_ncf.memory_judge_candidates_report.v1",
        "task_count": len(batches),
        "candidate_pair_count": sum(len(row["candidates"]) for row in batches),
        "candidates_per_task": sorted({len(row["candidates"]) for row in batches}),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "contains_final_labels": False,
        "judge_prompt_excludes": [
            "rule_seed_grade",
            "rule_seed_reason",
            "procedurally_valid_rule",
            "validity_reason_rule",
            "Dev/Test task, trajectory, reward and failure analysis",
        ],
        "uses_eval_data": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

