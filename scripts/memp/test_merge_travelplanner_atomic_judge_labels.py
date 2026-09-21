#!/usr/bin/env python3
"""Small regression test for train-only atomic-label merging."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/memp/merge_travelplanner_atomic_judge_labels.py"


def dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = {
            "family_id": "f",
            "operation_text": "operation",
            "parent_memory_id": 1,
            "subgoal_query": "query",
            "task_id": 0,
            "validation_or_test_used": False,
        }
        pairs = [
            {**base, "subgoal_id": "f:subgoal:route", "operation_id": "m:route:0", "model_split": "internal_train", "seed_grade": 2, "selection_role": "direct_anchor"},
            {**base, "subgoal_id": "f:subgoal:route", "operation_id": "m:closure:0", "model_split": "internal_train", "seed_grade": "uncertain", "selection_role": "same_family_complement"},
            {**base, "subgoal_id": "f:subgoal:route", "operation_id": "m:conflict:0", "model_split": "internal_train", "seed_grade": 0, "selection_role": "explicit_conflict"},
            {**base, "subgoal_id": "g:subgoal:route", "operation_id": "m:closure:0", "model_split": "internal_dev", "seed_grade": "uncertain", "selection_role": "same_family_complement"},
        ]
        labels = [
            {"subgoal_id": "f:subgoal:route", "operation_id": "m:route:0", "grade": 0, "confidence": 0.9, "evidence": "ignored", "reason": "ignored", "prompt_version": "test"},
            {"subgoal_id": "f:subgoal:route", "operation_id": "m:closure:0", "grade": 1, "confidence": 0.8, "evidence": "supports", "reason": "complement", "prompt_version": "test"},
            {"subgoal_id": "f:subgoal:route", "operation_id": "m:conflict:0", "grade": 2, "confidence": 0.9, "evidence": "ignored", "reason": "ignored", "prompt_version": "test"},
        ]
        pair_path, label_path, output = root / "pairs.jsonl", root / "labels.jsonl", root / "out"
        dump(pair_path, pairs); dump(label_path, labels)
        subprocess.run(
            ["python", str(SCRIPT), "--pairs", str(pair_path), "--judge-labels", str(label_path), "--output-dir", str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        merged = [json.loads(line) for line in (output / "pairs_all_with_provenance.jsonl").read_text().splitlines()]
        assert [row["final_grade"] for row in merged] == [2, 1, 0, "uncertain"]
        assert len((output / "train_pairs_resolved.jsonl").read_text().splitlines()) == 3
        assert len((output / "internal_dev_pairs_resolved.jsonl").read_text().splitlines()) == 0
        report = json.loads((output / "report.json").read_text())
        assert report["context_judgments_ignored"] == 2
        assert report["validation_or_test_used"] is False
        assert report["judge_confidence_used_as_training_weight"] is False
        assert report["resolved_duplicate_pair_count"] == 0
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
