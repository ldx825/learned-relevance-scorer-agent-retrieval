#!/usr/bin/env python3
"""Fail-closed audit of the zero-API graded-Judge pilot manifest."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from build_evidence_v1 import ROOT, load_jsonl


MANIFEST = ROOT / "data/skillbench_ncf/graded_judge_pilot_v4/judge_manifest.jsonl"
REPORT = (
    ROOT
    / "artifacts/skillbench_ncf/manifests/graded_judge_pilot_v4_report.json"
)


def main() -> int:
    rows = load_jsonl(MANIFEST)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if len(rows) != 30 or len({row["query_id"] for row in rows}) != 30:
        raise AssertionError("expected 30 unique queries")
    if Counter(row["internal_family_split"] for row in rows) != {
        "train": 24,
        "dev": 6,
    }:
        raise AssertionError("unexpected pilot split distribution")
    if any(row["labels"] is not None for row in rows):
        raise AssertionError("manifest contains labels before Judge")
    if any(row["api_submission_authorized"] for row in rows):
        raise AssertionError("manifest claims API authorization before approval")
    if any(row["official_skillsbench_task_used"] for row in rows):
        raise AssertionError("official SkillsBench task leakage")
    for row in rows:
        candidate_ids = [item["skill_id"] for item in row["candidates"]]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise AssertionError(f"duplicate candidate: {row['query_id']}")
        if not 10 <= len(candidate_ids) <= 20:
            raise AssertionError(f"bad candidate count: {row['query_id']}")
        if row["audit_source_skill_id"] not in candidate_ids:
            raise AssertionError(f"source anchor missing: {row['query_id']}")
    if report["api_calls_made"] != 0:
        raise AssertionError("zero-API report claims paid calls")
    print(
        "[PASS] 30-query graded-Judge pilot is fixed, split-safe, "
        "unlabeled, and free of official SkillsBench tasks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
