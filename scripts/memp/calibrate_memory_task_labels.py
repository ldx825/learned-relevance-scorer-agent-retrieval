#!/usr/bin/env python3
"""Calibrate raw Memory-Task Judge labels into final 2/1/0 pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def calibrated_grade(raw_grade: int, candidate: dict) -> tuple[int, str]:
    grade = int(raw_grade)
    action = "unchanged"
    seed_grade = int(candidate["rule_seed_grade"])
    seed_reason = candidate["rule_seed_reason"]
    if not candidate["procedurally_valid_rule"]:
        grade, action = 0, "force_0_known_procedural_violation"
    elif seed_grade == 2 and grade != 2:
        grade, action = 2, f"upgrade_{raw_grade}_to_2_explicit_exact_executable_match"
    elif seed_grade == 1 and grade == 2:
        grade, action = 1, "downgrade_2_to_1_explicit_partial_relation"
    elif seed_grade == 0 and grade != 0:
        grade, action = 0, f"downgrade_{raw_grade}_to_0_explicit_relation_mismatch_{seed_reason}"
    return grade, action


def deduplicate_grade1_roles(rows: list[dict]) -> None:
    """Keep one representative per task and deterministic support role.

    Memories with the same relation role are alternatives, not distinct
    workflow components.  We keep one representative for each role, but do
    not collapse different roles: object-side, destination-side, operation
    template and placement subprocedures may all remain Grade 1 together.
    """
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        if int(row["target_grade"]) != 1:
            continue
        grouped.setdefault((int(row["task_id"]), row["rule_seed_reason"]), []).append(row)
    for candidates in grouped.values():
        if len(candidates) <= 1:
            continue
        candidates.sort(key=lambda row: (-float(row["label_confidence"]), int(row["memory_id"])))
        for row in candidates[1:]:
            row["target_grade"] = 0
            row["target_relevance"] = 0.0
            row["label_confidence"] = min(float(row["label_confidence"]), 0.85)
            row["calibration_action"] = "downgrade_1_to_0_redundant_support_role"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    batches = load_jsonl(args.candidates)
    labels = {(int(row["task_id"]), int(row["memory_id"])): row for row in load_jsonl(args.labels)}
    output = []
    raw_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    missing = []
    uncertain = 0

    for batch in batches:
        for candidate in batch["candidates"]:
            key = (int(batch["task_id"]), int(candidate["memory_id"]))
            if key not in labels:
                missing.append(key)
                continue
            label = labels[key]
            raw_grade = label["grade"]
            raw_counts[str(raw_grade)] += 1
            if raw_grade == "uncertain":
                uncertain += 1
                continue
            grade = int(raw_grade)
            seed_grade = int(candidate["rule_seed_grade"])
            seed_reason = candidate["rule_seed_reason"]
            grade, action = calibrated_grade(grade, candidate)
            confidence = float(label["confidence"])
            if action != "unchanged":
                confidence = min(confidence, 0.85)
            action_counts[action] += 1
            target_counts[str(grade)] += 1
            output.append(
                {
                    "schema_version": "task_resource_ncf.memory_task_pair.v1",
                    "pair_id": f"{key[0]}::{key[1]}",
                    "task_id": key[0],
                    "memory_id": key[1],
                    "internal_split": batch["internal_split"],
                    "candidate_sources": candidate["candidate_sources"],
                    "raw_judge_grade": raw_grade,
                    "target_grade": grade,
                    "target_relevance": grade / 2.0,
                    "label_confidence": confidence,
                    "judge_evidence": {
                        field: label[field]
                        for field in (
                            "operation",
                            "object",
                            "cardinality",
                            "destination",
                            "procedural_validity",
                            "distinct_contribution",
                            "reason",
                        )
                    },
                    "rule_seed_grade": seed_grade,
                    "rule_seed_reason": seed_reason,
                    "calibration_action": action,
                    "label_source": "memory_llm_judge_plus_explicit_relation_calibration",
                    "uses_eval_data": False,
                }
            )

    deduplicate_grade1_roles(output)
    # Recompute counts after the set-level calibration above.
    target_counts = Counter(str(row["target_grade"]) for row in output)
    action_counts = Counter(row["calibration_action"] for row in output)
    output.sort(key=lambda row: (row["internal_split"], row["task_id"], row["memory_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pairs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output), encoding="utf-8")
    report = {
        "schema_version": "task_resource_ncf.memory_labeled_pairs_report.v1",
        "pair_count": len(output),
        "missing_label_count": len(missing),
        "uncertain_excluded_count": uncertain,
        "raw_judge_grade_counts": dict(sorted(raw_counts.items())),
        "target_grade_counts": dict(sorted(target_counts.items())),
        "calibration_action_counts": dict(sorted(action_counts.items())),
        "uses_eval_data": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
