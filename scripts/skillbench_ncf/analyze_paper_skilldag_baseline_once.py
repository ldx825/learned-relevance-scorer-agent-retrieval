#!/usr/bin/env python3
"""Analyze the final one-attempt MiniMax SkillDAG SkillsBench baseline.

The final 87-task view combines:
* 9 previously audited trials listed in reuse_manifest.json;
* 67 scored trials from the main job;
* 11 replacement trials from the infrastructure retry job.

The script never calls an API.  It only reads saved Harbor artifacts and the
official task-local ``environment/skills`` directories.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/SkillDAG_NCF/analysis"))

from score_skillsbench_gos import collect_trials  # noqa: E402


RUN_ROOT = ROOT / ".runtime/skillbench_ncf/eval/paper_skilldag_once"
MAIN_JOB = RUN_ROOT / "results/skillsbench-paper-skilldag-once"
RETRY_JOB = (
    ROOT
    / ".runtime/skillbench_ncf/eval/paper_skilldag_infra_retry_once/results"
    / "skillsbench-paper-skilldag-infra-retry-once"
)
MANIFEST = RUN_ROOT / "reuse_manifest.json"
TASK_ROOT = ROOT / ".runtime/skillbench_ncf/official/skillsbench-v1.0/tasks"
OUTPUT = RUN_ROOT / "final_failure_analysis.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def trial_dir_from_result(path: Path) -> Path:
    return path.parent


def command_strings(trial_dir: Path) -> list[str]:
    trajectory = trial_dir / "agent/trajectory.json"
    if not trajectory.exists():
        return []
    payload = load_json(trajectory)
    commands: list[str] = []
    for step in payload.get("steps") or []:
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            command = args.get("keystrokes")
            if isinstance(command, str) and command.strip():
                commands.append(command.strip())
    return commands


def shown_ids(commands: list[str]) -> set[str]:
    found: set[str] = set()
    for command in commands:
        for line in command.splitlines():
            if "skilldag show " not in line:
                continue
            suffix = line.split("skilldag show ", 1)[1]
            try:
                words = shlex.split(suffix)
            except ValueError:
                words = suffix.split()
            for word in words:
                if word.startswith("-") or word in {"&&", "||", ";", "|"}:
                    break
                found.add(word.strip())
    return found


def gold_ids(task_name: str) -> set[str]:
    skills = TASK_ROOT / task_name / "environment/skills"
    if not skills.is_dir():
        return set()
    return {path.name for path in skills.iterdir() if path.is_dir() and (path / "SKILL.md").exists()}


def query_count(commands: list[str]) -> int:
    total = 0
    for command in commands:
        for line in command.splitlines():
            if "skilldag graph search " not in line:
                continue
            suffix = line.split("skilldag graph search ", 1)[1]
            try:
                words = shlex.split(suffix)
            except ValueError:
                total += 1
                continue
            positional = []
            for word in words:
                if word.startswith("--"):
                    break
                positional.append(word)
            total += max(1, len(positional))
    return total


def selected_trials() -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    for item in load_json(MANIFEST)["accepted"]:
        path = ROOT / item["result_path"]
        result = load_json(path)
        selected[item["task_id"]] = {
            "task_name": item["task_id"],
            "reward": float(item.get("reward") or 0.0),
            "status": "scored",
            "exception_kind": (result.get("exception_info") or {}).get("exception_type"),
            "result_path": str(path),
        }

    for trial in collect_trials(MAIN_JOB):
        if trial.status == "scored":
            selected[trial.task_name] = asdict(trial)

    # Every retry result intentionally replaces the corresponding main failure.
    for trial in collect_trials(RETRY_JOB):
        selected[trial.task_name] = asdict(trial)

    if len(selected) != 87:
        raise RuntimeError(f"Expected 87 unique tasks, got {len(selected)}")
    return [selected[name] for name in sorted(selected)]


def main() -> int:
    rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    search_calls = query_total = edit_calls = 0

    for trial in selected_trials():
        task_name = trial["task_name"]
        result_path = Path(trial["result_path"])
        trial_dir = trial_dir_from_result(result_path)
        commands = command_strings(trial_dir)
        shown = shown_ids(commands)
        gold = gold_ids(task_name)
        shown_gold = shown & gold
        reward = trial.get("score_reward")
        if reward is None:
            reward = trial.get("reward")
        reward = float(reward or 0.0)

        this_search_calls = sum(command.count("skilldag graph search ") for command in commands)
        this_edit_calls = sum(command.count("skilldag graph edit-edge ") for command in commands)
        this_query_count = query_count(commands)
        search_calls += this_search_calls
        edit_calls += this_edit_calls
        query_total += this_query_count

        if trial["status"] == "infra_excluded":
            category = "infrastructure"
        elif reward == 1.0:
            category = "success"
        elif not this_search_calls:
            category = "protocol_or_runner_before_search"
        elif gold and not shown_gold:
            category = "no_gold_shown"
        elif gold and shown_gold == gold:
            category = "all_gold_shown_execution_failure"
        else:
            category = "partial_gold_shown_execution_failure"
        category_counts[category] += 1

        rows.append(
            {
                "task_name": task_name,
                "reward": reward,
                "status": trial["status"],
                "exception_kind": trial.get("exception_kind"),
                "category": category,
                "gold_ids": sorted(gold),
                "shown_ids": sorted(shown),
                "shown_gold_ids": sorted(shown_gold),
                "missing_gold_ids": sorted(gold - shown_gold),
                "search_calls": this_search_calls,
                "search_queries": this_query_count,
                "edit_edge_calls": this_edit_calls,
                "result_path": str(result_path.relative_to(ROOT)),
            }
        )

    valid = [row for row in rows if row["category"] != "infrastructure"]
    reward_sum = sum(row["reward"] for row in valid)
    payload = {
        "schema_version": "skillbench.paper_skilldag.single_run_failure_analysis.v1",
        "protocol": {
            "task_count": 87,
            "attempts_per_task": 1,
            "model": "MiniMax-M2.7 via Yunwu OpenAI-compatible endpoint",
            "skill_scale": 1000,
            "note": "Single-run diagnostic; the paper averages two attempts for SkillsBench task reward.",
        },
        "official_reward": {
            "reward_sum": reward_sum,
            "valid_tasks": len(valid),
            "infrastructure_excluded": 87 - len(valid),
            "valid_only_r_percent": 100.0 * reward_sum / len(valid),
            "all_87_zero_fill_r_percent": 100.0 * reward_sum / 87,
            "paper_minimax_skilldag_r_percent": 27.3,
        },
        "internal_diagnostic": {
            "definition": "Gold shown means the agent explicitly called skilldag show for an official task-local skill id.",
            "category_counts": dict(category_counts),
            "search_calls": search_calls,
            "search_queries": query_total,
            "edit_edge_calls": edit_calls,
            "warning": "These are internal causal diagnostics, not the paper's Ret@K/Ret@1/MRR metrics.",
        },
        "category_tasks": {
            category: [row["task_name"] for row in rows if row["category"] == category]
            for category in sorted(category_counts)
        },
        "tasks": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
