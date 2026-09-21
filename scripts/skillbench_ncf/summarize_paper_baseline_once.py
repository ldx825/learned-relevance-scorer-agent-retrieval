#!/usr/bin/env python3
"""Combine the 78 new and 9 audited reused SkillsBench baseline trials."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/SkillDAG_NCF/analysis"))

from score_skillsbench_gos import collect_trials  # noqa: E402


RUN_ROOT = ROOT / ".runtime/skillbench_ncf/eval/paper_skilldag_once"
JOB = RUN_ROOT / "results/skillsbench-paper-skilldag-once"
MANIFEST = RUN_ROOT / "reuse_manifest.json"
OUTPUT = RUN_ROOT / "combined_summary.json"


def main() -> int:
    trials = collect_trials(JOB)
    scored = [trial for trial in trials if trial.status == "scored"]
    infra = [trial for trial in trials if trial.status == "infra_excluded"]
    reused = json.loads(MANIFEST.read_text(encoding="utf-8"))["accepted"]
    new_reward = sum(float(trial.score_reward or 0.0) for trial in scored)
    reused_reward = sum(float(item.get("reward") or 0.0) for item in reused)
    scored_total = len(scored) + len(reused)
    reward_total = new_reward + reused_reward
    payload = {
        "paper_snapshot_tasks": 87,
        "new_trials_finished": len(trials),
        "new_trials_scored": len(scored),
        "reused_trials_scored": len(reused),
        "infra_excluded_pending_rerun": len(infra),
        "reward_sum": reward_total,
        "interim_scored_only_r_percent": 100.0 * reward_total / scored_total,
        "nonfinal_all87_zero_fill_r_percent": 100.0 * reward_total / 87,
        "exception_kinds": dict(Counter(trial.exception_kind for trial in trials if trial.exception_kind)),
        "warning": "Neither R value is final. Rerun the 11 infrastructure failures before the complete 87-task comparison.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
