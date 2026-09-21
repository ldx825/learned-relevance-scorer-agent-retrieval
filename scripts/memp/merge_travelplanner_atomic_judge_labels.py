#!/usr/bin/env python3
"""Merge train-only GPT Judge decisions into atomic TravelPlanner pairs.

Only seed ``uncertain`` pairs in ``internal_train`` may be replaced.  All
deterministic anchors, complements, negatives, and explicit conflicts remain
locked.  Internal-dev uncertain pairs stay unresolved and are excluded from
the resolved evaluation file so the Judge never supplies dev supervision.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["subgoal_id"]), str(row["operation_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--judge-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    labels = read_jsonl(args.judge_labels)
    label_by_key = {pair_key(row): row for row in labels}
    if len(label_by_key) != len(labels):
        raise SystemExit("duplicate Judge labels")

    expected = {
        pair_key(row)
        for row in pairs
        if row["model_split"] == "internal_train" and row["seed_grade"] == "uncertain"
    }
    missing = expected - set(label_by_key)
    if missing:
        raise SystemExit(f"missing {len(missing)} train uncertain Judge labels")

    merged: list[dict[str, Any]] = []
    resolved_train: list[dict[str, Any]] = []
    resolved_dev: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    grade_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    judge_grade_by_required_key: defaultdict[str, Counter[str]] = defaultdict(Counter)
    judge_confidence_by_grade: defaultdict[str, list[float]] = defaultdict(list)

    for pair in pairs:
        row = dict(pair)
        key = pair_key(row)
        if row["model_split"] == "internal_train" and row["seed_grade"] == "uncertain":
            judged = label_by_key[key]
            grade = judged["grade"]
            if grade == "uncertain":
                row.update(
                    final_grade="uncertain",
                    label_source="gpt4o_train_only_unresolved",
                    judge_confidence=judged["confidence"],
                    judge_evidence=judged["evidence"],
                    judge_reason=judged["reason"],
                    judge_prompt_version=judged["prompt_version"],
                )
            else:
                row.update(
                    final_grade=int(grade),
                    label_source="gpt4o_train_only_uncertain_resolution",
                    judge_confidence=judged["confidence"],
                    judge_evidence=judged["evidence"],
                    judge_reason=judged["reason"],
                    judge_prompt_version=judged["prompt_version"],
                )
            transition_counts[f"uncertain->{grade}"] += 1
            required_key = str(row["subgoal_id"]).rsplit(":", 1)[-1]
            judge_grade_by_required_key[required_key][str(grade)] += 1
            judge_confidence_by_grade[str(grade)].append(float(judged["confidence"]))
        else:
            row.update(final_grade=row["seed_grade"], label_source="deterministic_locked")

        # These invariants must never depend on an LLM judgment.
        if row["selection_role"] == "direct_anchor" and row["final_grade"] != 2:
            raise SystemExit(f"direct anchor was not locked at Grade 2: {key}")
        if row["selection_role"] == "explicit_conflict" and row["final_grade"] != 0:
            raise SystemExit(f"explicit conflict was not locked at Grade 0: {key}")

        merged.append(row)
        grade_by_split[str(row["model_split"])][str(row["final_grade"])] += 1
        if row["final_grade"] != "uncertain":
            (resolved_train if row["model_split"] == "internal_train" else resolved_dev).append(row)

    unexpected = set(label_by_key) - expected
    # Extra judgments are expected because each compact Judge request includes
    # locked anchors/negatives as context.  They are deliberately never merged.
    resolved = resolved_train + resolved_dev
    group_grades: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    for row in resolved:
        group_grades[(str(row["model_split"]), str(row["subgoal_id"]))].add(int(row["final_grade"]))
    report = {
        "schema_version": "memp.travelplanner.atomic_memory_labels.v1",
        "input_pair_count": len(pairs),
        "judge_label_count": len(labels),
        "train_uncertain_expected": len(expected),
        "train_uncertain_resolved": sum(
            count for name, count in transition_counts.items() if name != "uncertain->uncertain"
        ),
        "train_uncertain_still_unresolved": transition_counts["uncertain->uncertain"],
        "context_judgments_ignored": len(unexpected),
        "transition_counts": dict(sorted(transition_counts.items())),
        "final_grade_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(grade_by_split.items())
        },
        "judge_grade_by_required_operation": {
            key: dict(sorted(counts.items())) for key, counts in sorted(judge_grade_by_required_key.items())
        },
        "judge_confidence_mean_by_grade": {
            grade: round(sum(values) / len(values), 4)
            for grade, values in sorted(judge_confidence_by_grade.items())
            if values
        },
        "resolved_train_pair_count": len(resolved_train),
        "resolved_dev_pair_count": len(resolved_dev),
        "resolved_duplicate_pair_count": len(resolved) - len({pair_key(row) for row in resolved}),
        "resolved_groups_without_grade2": sum(2 not in grades for grades in group_grades.values()),
        "resolved_groups_without_grade0": sum(0 not in grades for grades in group_grades.values()),
        "deterministic_labels_locked": True,
        "judge_confidence_used_as_training_weight": False,
        "validation_or_test_used": False,
        "audit_pass": not missing,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "pairs_all_with_provenance.jsonl", merged)
    write_jsonl(args.output_dir / "train_pairs_resolved.jsonl", resolved_train)
    write_jsonl(args.output_dir / "internal_dev_pairs_resolved.jsonl", resolved_dev)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
