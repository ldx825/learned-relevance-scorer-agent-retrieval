#!/usr/bin/env python3
"""Paired, API-free analysis of ALFWorld SkillDAG/NCF online runs."""

from __future__ import annotations

import argparse
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "src/SkillDAG_NCF/results/alfworld"
DEFAULT_BASELINE = RESULTS_ROOT / "pilot10_skilldag_yunwu_once_20260727"
DEFAULT_OLD_NCF = RESULTS_ROOT / "pilot10_skilldag_ncf_yunwu_once_20260727"
DEFAULT_V3 = RESULTS_ROOT / "failed4_skilldag_ncf_v3_yunwu_once_20260729"
DEFAULT_OUTPUT = (
    ROOT / ".runtime/skilldag_ncf_v3/reports/v3_online_paired_analysis.json"
)
DEFAULT_PHASE_REPORT = (
    ROOT / ".runtime/skilldag_ncf_v3/reports/phase_retrieval_comparison.json"
)

STATE_ACTION = {
    "pick_clean_then_place_in_recep": "clean",
    "pick_cool_then_place_in_recep": "cool",
    "pick_heat_then_place_in_recep": "heat",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_arm(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for result_path in sorted(path.glob("idx_*.json")):
        idx = int(result_path.stem.split("_", 1)[1])
        rows[idx] = load_json(result_path)
    return rows


def task_type(row: dict[str, Any]) -> str:
    return str(row["name"]).split("/", 1)[0].split("-", 1)[0]


def assistant_actions(row: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    decoder = json.JSONDecoder()
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        payload = None
        for start, char in enumerate(content):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(content[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        action = payload.get("action") if isinstance(payload, dict) else None
        if isinstance(action, str) and action.strip():
            actions.append(action.strip().lower())
    return actions


def shown_skills(row: dict[str, Any]) -> list[str]:
    skills: set[str] = set()
    for item in row.get("cli_invocations") or []:
        try:
            tokens = shlex.split(str(item.get("command", "")))
        except ValueError:
            continue
        try:
            show_index = tokens.index("show")
        except ValueError:
            continue
        if show_index == 0 or tokens[show_index - 1] != "skilldag":
            continue
        for token in tokens[show_index + 1 :]:
            if token.startswith("-"):
                break
            skills.add(token)
    return sorted(skills)


def observation_stats(row: dict[str, Any]) -> dict[str, int]:
    observations = [
        str(message.get("content", "")).lower()
        for message in row.get("messages", [])
        if message.get("role") == "user"
    ]
    return {
        "nothing_happens": sum("nothing happens" in text for text in observations),
        "not_found": sum("not found" in text for text in observations),
    }


def token_stats(row: dict[str, Any]) -> dict[str, int]:
    usage = row.get("token_usage") or {}
    prompt = int(usage.get("prompt") or 0)
    completion = int(usage.get("completion") or 0)
    return {
        "prompt": prompt,
        "completion": completion,
        "billable_noncache": prompt + completion,
        "cache_read": int(usage.get("cache_read") or 0),
    }


def command_stats(row: dict[str, Any]) -> dict[str, int]:
    invocations = row.get("cli_invocations") or []
    terminal_words = ("done", "success", "complete", "finish", "exit", "stop", "final")
    return {
        "cli_errors": sum(int(item.get("rc", 0)) != 0 for item in invocations),
        "search_calls": sum(
            "skilldag graph search" in str(item.get("command", ""))
            for item in invocations
        ),
        "show_calls": sum(
            "skilldag show " in str(item.get("command", "")) for item in invocations
        ),
        "terminal_echo_calls": sum(
            str(item.get("command", "")).strip().lower().startswith("echo ")
            and any(
                word in str(item.get("command", "")).lower()
                for word in terminal_words
            )
            for item in invocations
        ),
    }


def summarize_run(row: dict[str, Any]) -> dict[str, Any]:
    actions = assistant_actions(row)
    counts = Counter(actions)
    kind = task_type(row)
    required = STATE_ACTION.get(kind)
    tokens = token_stats(row)
    return {
        "success": bool(row.get("reward")),
        "steps": int(row.get("steps") or 0),
        "turns": int(row.get("turns") or 0),
        "cli_calls": int(row.get("cli_calls") or 0),
        "cli_budget_exhausted": bool(row.get("cli_budget_exhausted")),
        "runtime_seconds": float(row.get("agent_runtime_seconds") or 0.0),
        "loaded_skills_reported": sorted(set(row.get("loaded_skills") or [])),
        "shown_skills": shown_skills(row),
        "actions": actions,
        "unique_action_count": len(counts),
        "max_identical_action_count": max(counts.values(), default=0),
        "required_state_action": required,
        "required_state_action_present": (
            None
            if required is None
            else any(action.startswith(required + " ") for action in actions)
        ),
        **tokens,
        **observation_stats(row),
        **command_stats(row),
    }


def phase_recommendations(row: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for phase in row.get("phase_recommendations") or []:
        output.append(
            {
                "phase": phase.get("phase"),
                "phase_group": phase.get("phase_group"),
                "query": phase.get("query"),
                "top3": [
                    match.get("skill_id")
                    for match in (phase.get("matches") or [])[:3]
                    if match.get("skill_id")
                ],
            }
        )
    return output


def means(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = (
        "steps",
        "turns",
        "cli_calls",
        "runtime_seconds",
        "prompt",
        "completion",
        "billable_noncache",
        "cache_read",
        "cli_errors",
        "search_calls",
        "show_calls",
        "terminal_echo_calls",
        "nothing_happens",
        "max_identical_action_count",
    )
    return {key: fmean(float(row[key]) for row in rows) for key in keys}


def arm_summary(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    normalized = [summarize_run(row) for row in rows.values()]
    successful = [row for row in normalized if row["success"]]
    return {
        "n_tasks": len(normalized),
        "n_success": len(successful),
        "success_rate": len(successful) / len(normalized),
        "all_task_means": means(normalized),
        "successful_task_means": means(successful),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--old-ncf", type=Path, default=DEFAULT_OLD_NCF)
    parser.add_argument("--v3", type=Path, default=DEFAULT_V3)
    parser.add_argument("--phase-report", type=Path, default=DEFAULT_PHASE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    arms = {
        "SkillDAG": load_arm(args.baseline),
        "Old_NCF": load_arm(args.old_ncf),
        "V3": load_arm(args.v3),
    }
    expected = set(range(140))
    coverage = {name: sorted(expected - set(rows)) for name, rows in arms.items()}
    if any(coverage.values()):
        raise RuntimeError(f"incomplete 140-task arm(s): {coverage}")
    task_alignment = {
        idx: {
            (rows[idx].get("name"), rows[idx].get("query"))
            for rows in arms.values()
        }
        for idx in expected
    }
    misaligned = [idx for idx, values in task_alignment.items() if len(values) != 1]
    if misaligned:
        raise RuntimeError(f"task/query mismatch across arms: {misaligned}")

    detailed: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    for idx in sorted(expected):
        baseline = summarize_run(arms["SkillDAG"][idx])
        old_ncf = summarize_run(arms["Old_NCF"][idx])
        v3 = summarize_run(arms["V3"][idx])
        if baseline["success"] and v3["success"]:
            category = "both_success"
        elif baseline["success"]:
            category = "baseline_only"
        elif v3["success"]:
            category = "v3_only"
        else:
            category = "both_fail"
        category_counts[category] += 1
        row = {
            "idx": idx,
            "task_id": arms["V3"][idx]["name"],
            "task_type": task_type(arms["V3"][idx]),
            "query": arms["V3"][idx]["query"],
            "category": category,
            "SkillDAG": baseline,
            "Old_NCF": old_ncf,
            "V3": v3,
            "v3_phase_recommendations": phase_recommendations(arms["V3"][idx]),
        }
        detailed.append(row)
        by_type[row["task_type"]].append(row)

    type_summary: dict[str, Any] = {}
    for kind, rows in sorted(by_type.items()):
        n = len(rows)
        baseline_success = sum(row["SkillDAG"]["success"] for row in rows)
        old_success = sum(row["Old_NCF"]["success"] for row in rows)
        v3_success = sum(row["V3"]["success"] for row in rows)
        type_summary[kind] = {
            "n_tasks": n,
            "SkillDAG": {
                "n_success": baseline_success,
                "success_rate": baseline_success / n,
            },
            "Old_NCF": {
                "n_success": old_success,
                "success_rate": old_success / n,
            },
            "V3": {"n_success": v3_success, "success_rate": v3_success / n},
            "net_v3_vs_skilldag": v3_success - baseline_success,
            "categories": dict(
                sorted(Counter(row["category"] for row in rows).items())
            ),
        }

    category_summary = {}
    for category in (
        "v3_only",
        "baseline_only",
        "both_fail",
        "both_success",
    ):
        rows = [row for row in detailed if row["category"] == category]
        category_summary[category] = {
            "n_tasks": len(rows),
            "indices": [row["idx"] for row in rows],
            "task_types": dict(
                sorted(Counter(row["task_type"] for row in rows).items())
            ),
            "SkillDAG_means": means([row["SkillDAG"] for row in rows]),
            "V3_means": means([row["V3"] for row in rows]),
            "state_task_missing_required_action": {
                "SkillDAG": [
                    row["idx"]
                    for row in rows
                    if row["SkillDAG"]["required_state_action_present"] is False
                ],
                "V3": [
                    row["idx"]
                    for row in rows
                    if row["V3"]["required_state_action_present"] is False
                ],
            },
            "v3_step_budget_failures": [
                row["idx"]
                for row in rows
                if not row["V3"]["success"] and row["V3"]["steps"] >= 30
            ],
            "v3_cli_budget_failures": [
                row["idx"]
                for row in rows
                if not row["V3"]["success"]
                and row["V3"]["cli_budget_exhausted"]
            ],
        }

    phase_by_category: dict[str, Any] = {}
    if args.phase_report.exists():
        phase_report = load_json(args.phase_report)
        category_by_idx = {row["idx"]: row["category"] for row in detailed}
        phase_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in phase_report.get("rows", []):
            category = category_by_idx[int(row["idx"])]
            gold = set(row["gold"])
            cosine = row["skilldag_top_k"][:3]
            ncf = row["skilldag_ncf_top_k"][:3]
            phase_counts[category]["n_phases"] += 1
            phase_counts[category]["cosine_hit3"] += bool(gold & set(cosine))
            phase_counts[category]["ncf_hit3"] += bool(gold & set(ncf))
        for category, counts in sorted(phase_counts.items()):
            n = counts["n_phases"]
            phase_by_category[category] = {
                "n_phases": n,
                "SkillDAG_Phase_Hit@3": counts["cosine_hit3"] / n,
                "V3_Phase_Hit@3": counts["ncf_hit3"] / n,
            }

    v3_failures = [row for row in detailed if not row["V3"]["success"]]
    output = {
        "schema_version": "skilldag_ncf.v3.online_paired_analysis.v1",
        "definition": {
            "unit": "paired ALFWorld valid_seen task index",
            "api_calls_made": 0,
            "success": "boolean reward in idx_<n>.json",
            "token_cost": "prompt + completion; cache_read reported separately",
            "caution": (
                "Single stochastic run per arm: paired flips are associations, "
                "not task-level causal proof."
            ),
        },
        "inputs": {
            "SkillDAG": str(args.baseline),
            "Old_NCF": str(args.old_ncf),
            "V3": str(args.v3),
            "phase_report": str(args.phase_report),
        },
        "integrity": {
            "expected_task_indices": "0..139",
            "task_query_alignment": True,
            "max_environment_steps": {
                name: max(int(row.get("steps") or 0) for row in rows.values())
                for name, rows in arms.items()
            },
            "agent_edit_counts": {
                name: sum(
                    len(row.get("agent_edits") or []) for row in rows.values()
                )
                for name, rows in arms.items()
            },
        },
        "arms": {name: arm_summary(rows) for name, rows in arms.items()},
        "paired_v3_vs_skilldag": {
            "categories": dict(sorted(category_counts.items())),
            "net_success_gain": (
                category_counts["v3_only"] - category_counts["baseline_only"]
            ),
            "success_rate_gain_points": (
                arms_success_rate(arms["V3"])
                - arms_success_rate(arms["SkillDAG"])
            )
            * 100,
        },
        "by_task_type": type_summary,
        "by_category": category_summary,
        "phase_retrieval_by_outcome": phase_by_category,
        "v3_failure_diagnostics": {
            "n_failures": len(v3_failures),
            "missing_required_state_action": [
                row["idx"]
                for row in v3_failures
                if row["V3"]["required_state_action_present"] is False
            ],
            "step_budget_exhausted": [
                row["idx"] for row in v3_failures if row["V3"]["steps"] >= 30
            ],
            "cli_budget_exhausted": [
                row["idx"]
                for row in v3_failures
                if row["V3"]["cli_budget_exhausted"]
            ],
            "terminal_echo_loop": [
                row["idx"]
                for row in v3_failures
                if row["V3"]["terminal_echo_calls"] >= 5
            ],
        },
        "rows": detailed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["arms"], ensure_ascii=False, indent=2))
    print(json.dumps(output["paired_v3_vs_skilldag"], ensure_ascii=False, indent=2))
    print(json.dumps(output["by_task_type"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0


def arms_success_rate(rows: dict[int, dict[str, Any]]) -> float:
    return sum(bool(row.get("reward")) for row in rows.values()) / len(rows)


if __name__ == "__main__":
    raise SystemExit(main())
