#!/usr/bin/env python3
"""Freeze the infrastructure-only retry set from the 87-task baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/SkillDAG_NCF/analysis"))

from score_skillsbench_gos import collect_trials  # noqa: E402


SOURCE = ROOT / ".runtime/skillbench_ncf/eval/paper_skilldag_once/results/skillsbench-paper-skilldag-once"
OUTPUT = ROOT / ".runtime/skillbench_ncf/eval/paper_skilldag_infra_retry_once/retry_manifest.json"


def main() -> int:
    trials = collect_trials(SOURCE)
    retry = [trial for trial in trials if trial.status == "infra_excluded"]
    if len(trials) != 78:
        raise AssertionError(f"expected 78 new baseline trials, got {len(trials)}")
    if len(retry) != 11:
        raise AssertionError(f"expected 11 infrastructure retries, got {len(retry)}")
    payload = {
        "schema_version": "skillbench_ncf.paper_infra_retry.v1",
        "source_job": str(SOURCE.relative_to(ROOT)),
        "retry_count": len(retry),
        "task_ids": sorted(trial.task_name for trial in retry),
        "records": [
            {
                "task_id": trial.task_name,
                "exception_type": trial.exception_type,
                "exception_kind": trial.exception_kind,
                "exception_message": trial.exception_message,
                "source_result": trial.result_path,
            }
            for trial in sorted(retry, key=lambda item: item.task_name)
        ],
        "contract": "Only infrastructure-excluded trials are rerun; no scored task is repeated.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"retry_count": len(retry), "task_ids": payload["task_ids"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
