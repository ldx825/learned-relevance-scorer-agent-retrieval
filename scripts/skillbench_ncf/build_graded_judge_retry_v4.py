#!/usr/bin/env python3
"""Build a retry manifest containing only failed full-Judge queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_evidence_v1 import ROOT, dump_json, dump_jsonl, load_jsonl


FULL_MANIFEST = ROOT / "data/skillbench_ncf/graded_judge_full_v4/judge_manifest.jsonl"
FAILURES = ROOT / "data/skillbench_ncf/graded_judge_full_v4/judgments/failures.jsonl"
RETRY_MANIFEST = (
    ROOT / "data/skillbench_ncf/graded_judge_full_v4/retry/judge_manifest.jsonl"
)
REPORT = (
    ROOT
    / "artifacts/skillbench_ncf/manifests/graded_judge_full_v4_retry_manifest.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", type=Path, default=FAILURES)
    parser.add_argument("--output", type=Path, default=RETRY_MANIFEST)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    args.failures = (
        args.failures
        if args.failures.is_absolute()
        else (ROOT / args.failures).resolve()
    )
    args.output = (
        args.output if args.output.is_absolute() else (ROOT / args.output).resolve()
    )
    args.report = (
        args.report if args.report.is_absolute() else (ROOT / args.report).resolve()
    )
    rows = load_jsonl(FULL_MANIFEST)
    failures = load_jsonl(args.failures)
    failed_ids = {row["query_id"] for row in failures}
    retry = [row for row in rows if row["query_id"] in failed_ids]
    if len(retry) != len(failures) or len(failed_ids) != len(failures):
        raise AssertionError("failure IDs are not unique or complete")
    dump_jsonl(args.output, retry)
    report = {
        "schema_version": "skillbench_ncf.graded_judge_retry_manifest.v4",
        "status": "complete",
        "retry_queries": len(retry),
        "official_skillsbench_tasks_used": False,
        "reason": "first-pass response was not structurally parseable",
        "semantic_prompt_changed": False,
        "private_manifest": str(args.output.relative_to(ROOT)),
    }
    dump_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
