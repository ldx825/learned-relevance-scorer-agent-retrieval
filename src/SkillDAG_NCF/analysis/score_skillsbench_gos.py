#!/usr/bin/env python3
"""GoS-aligned scoring for Harbor/SkillsBench runs.

Paper-aligned accounting:
- R is mean verifier reward over scored trials, reported as percent.
- AgentTimeoutError after substantive execution stays in the denominator.
- Model/protocol failures stay in the denominator with reward 0 if no verifier
  reward is available.
- Startup/build infrastructure failures are excluded from the R
  denominator and reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


INFRA_EXCEPTION_TYPES = {
    "AgentSetupTimeoutError",
    "EnvironmentStartTimeoutError",
}

INFRA_MESSAGE_MARKERS = (
    "SkillDAG runtime failed to start",
    "Docker compose command failed",
    "docker compose command failed",
    "dockerfile parse error",
    "failed to solve:",
    "failed to resolve reference",
    "connect: connection refused",
    "RPC failed;",
    "fetch-pack: unexpected disconnect",
    "fatal: early EOF",
)


@dataclass
class TrialScore:
    trial_name: str
    task_name: str
    reward: float | None
    score_reward: float | None
    status: str
    exception_type: str | None
    exception_kind: str | None
    exception_message: str | None
    result_path: str | None


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def reward_from_result(result: dict[str, Any] | None) -> float | None:
    if not result:
        return None
    verifier_result = result.get("verifier_result") or {}
    rewards = verifier_result.get("rewards")
    if isinstance(rewards, dict):
        return as_float(rewards.get("reward"))
    return as_float(rewards)


def reward_from_files(trial_dir: Path) -> float | None:
    text_path = trial_dir / "verifier" / "reward.txt"
    if text_path.exists():
        try:
            return as_float(text_path.read_text().strip())
        except OSError:
            return None

    json_path = trial_dir / "verifier" / "reward.json"
    data = load_json(json_path) if json_path.exists() else None
    if isinstance(data, dict):
        return as_float(data.get("reward"))
    return None


def exception_info(result: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not result:
        return None, None
    info = result.get("exception_info")
    if not isinstance(info, dict):
        return None, None
    exc_type = info.get("exception_type")
    exc_msg = info.get("exception_message")
    return (
        exc_type if isinstance(exc_type, str) else None,
        exc_msg if isinstance(exc_msg, str) else None,
    )


def exception_kind(exc_type: str | None, message: str | None) -> str | None:
    text = message or ""
    if exc_type == "AgentTimeoutError":
        return "agent_timeout"
    if exc_type == "OutputLengthExceededError":
        return "output_length_exceeded"
    if exc_type in INFRA_EXCEPTION_TYPES:
        return "startup_timeout"
    if "SkillDAG runtime failed to start" in text:
        return "runtime_startup_failed"
    if "dockerfile parse error" in text.lower():
        return "dockerfile_parse_error"
    if "Docker compose command failed" in text or "docker compose command failed" in text:
        return "docker_build_failed"
    if exc_type == "RuntimeError":
        return "runtime_error"
    return exc_type


def is_infra_failure(exc_type: str | None, message: str | None) -> bool:
    if exc_type in INFRA_EXCEPTION_TYPES:
        return True
    text = message or ""
    return any(marker in text for marker in INFRA_MESSAGE_MARKERS)


def base_task_name(trial_dir: Path) -> str:
    name = trial_dir.name
    if "__" in name:
        return name.rsplit("__", 1)[0]
    return name


def is_trial_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "result.json").exists()
        or (path / "config.json").exists()
        or (path / "exception.txt").exists()
        or (path / "verifier").exists()
    )


def resolve_job_dir(path: Path, job_name: str | None = None) -> Path:
    path = path.expanduser().resolve()
    if job_name:
        candidate = path / job_name
        if candidate.is_dir():
            return candidate

    if (path / "config.json").exists() and any(is_trial_dir(p) for p in path.iterdir() if p.is_dir()):
        return path

    candidates = [
        p for p in path.iterdir()
        if p.is_dir() and (p / "config.json").exists() and any(is_trial_dir(c) for c in p.iterdir() if c.is_dir())
    ] if path.is_dir() else []
    if not candidates:
        return path
    return max(candidates, key=lambda p: p.stat().st_mtime)


def expected_trial_count(job_dir: Path, override: int | None) -> tuple[int | None, int | None]:
    if override is not None:
        return override, None

    config = load_json(job_dir / "config.json")
    if not config:
        return None, None

    attempts = int(config.get("n_attempts") or 1)
    task_names: set[str] = set()
    for dataset in config.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        include = dataset.get("task_names")
        exclude = set(dataset.get("exclude_task_names") or [])
        if include:
            task_names.update(str(name) for name in include if str(name) not in exclude)
            continue
        dataset_path = dataset.get("path")
        if not dataset_path:
            continue
        root = Path(str(dataset_path)).expanduser()
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name not in exclude:
                task_names.add(child.name)

    if not task_names:
        return None, attempts
    return len(task_names) * attempts, attempts


def score_trial(trial_dir: Path) -> TrialScore:
    result_path = trial_dir / "result.json"
    result = load_json(result_path) if result_path.exists() else None
    task_name = base_task_name(trial_dir)
    trial_name = trial_dir.name

    if result:
        trial_name = str(result.get("trial_name") or trial_name)
        task_name = str(result.get("task_name") or task_name)

    reward = reward_from_result(result)
    if reward is None:
        reward = reward_from_files(trial_dir)

    exc_type, exc_msg = exception_info(result)
    kind = exception_kind(exc_type, exc_msg)

    if result is None:
        status = "pending"
        score_reward = None
    elif is_infra_failure(exc_type, exc_msg):
        status = "infra_excluded"
        score_reward = None
    else:
        status = "scored"
        score_reward = reward if reward is not None else 0.0

    return TrialScore(
        trial_name=trial_name,
        task_name=task_name,
        reward=reward,
        score_reward=score_reward,
        status=status,
        exception_type=exc_type,
        exception_kind=kind,
        exception_message=exc_msg,
        result_path=str(result_path) if result_path.exists() else None,
    )


def collect_trials(job_dir: Path) -> list[TrialScore]:
    trials = []
    for child in sorted(job_dir.iterdir() if job_dir.exists() else [], key=lambda p: p.name):
        if is_trial_dir(child):
            trials.append(score_trial(child))
    return trials


def build_summary(job_dir: Path, expected_total: int | None) -> dict[str, Any]:
    trials = collect_trials(job_dir)
    scored = [t for t in trials if t.status == "scored"]
    infra = [t for t in trials if t.status == "infra_excluded"]
    pending_dirs = [t for t in trials if t.status == "pending"]
    finished = len(scored) + len(infra)
    expected = expected_total
    if expected is None:
        expected = finished + len(pending_dirs) if trials else None

    reward_sum = sum(float(t.score_reward or 0.0) for t in scored)
    score_pct = (100.0 * reward_sum / len(scored)) if scored else None
    pass_count = sum(1 for t in scored if t.score_reward == 1.0)
    fractional_count = sum(1 for t in scored if t.score_reward not in (0.0, 1.0))
    zero_count = sum(1 for t in scored if t.score_reward == 0.0)
    exception_counts = Counter(t.exception_kind or t.exception_type for t in trials if t.exception_type)
    infra_counts = Counter(t.exception_kind or t.exception_type for t in infra if t.exception_type)
    pending_missing = max((expected or 0) - finished, 0) if expected is not None else None

    return {
        "job_dir": str(job_dir),
        "expected_trials": expected,
        "finished_trials": finished,
        "scored_trials": len(scored),
        "infra_excluded_trials": len(infra),
        "pending_or_running_trials": pending_missing,
        "pending_dirs_without_result": len(pending_dirs),
        "reward_sum": reward_sum,
        "score_R_percent": score_pct,
        "exact_pass": pass_count,
        "exact_fail": len(scored) - pass_count,
        "fractional_reward": fractional_count,
        "zero_reward": zero_count,
        "exception_counts": dict(sorted(exception_counts.items())),
        "infra_excluded_counts": dict(sorted(infra_counts.items())),
        "trials": [asdict(t) for t in trials],
    }


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{count} {name}" for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def print_text(summary: dict[str, Any]) -> None:
    expected = summary["expected_trials"]
    finished = summary["finished_trials"]
    scored = summary["scored_trials"]
    infra = summary["infra_excluded_trials"]
    pending = summary["pending_or_running_trials"]
    score = summary["score_R_percent"]
    score_text = "NA" if score is None else f"{score:.1f}%"

    denom = f"/{expected}" if expected is not None else ""
    print("GoS-aligned SkillsBench scoring")
    print(f"job_dir: {summary['job_dir']}")
    print(f"finished: {finished}{denom} (scored={scored}, infra_excluded={infra}, pending={pending})")
    print(
        "R: "
        f"{score_text} "
        f"(reward_sum={summary['reward_sum']:.6g}, denominator={scored})"
    )
    print(
        "exact pass/fail: "
        f"{summary['exact_pass']}/{scored} pass, {summary['exact_fail']} fail "
        f"(fractional={summary['fractional_reward']}, zero={summary['zero_reward']})"
    )
    print(f"exceptions: {sum(summary['exception_counts'].values())} ({format_counts(summary['exception_counts'])})")
    print(f"infra excluded: {infra} ({format_counts(summary['infra_excluded_counts'])})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Harbor job dir or parent jobs dir")
    parser.add_argument("--job-name", help="Job subdirectory name when path is a parent jobs dir")
    parser.add_argument("--total", type=int, help="Override expected trial count")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    parser.add_argument("--list-infra", action="store_true", help="List infra-excluded trials after the summary")
    args = parser.parse_args(argv)

    job_dir = resolve_job_dir(args.path, args.job_name)
    expected_total, _attempts = expected_trial_count(job_dir, args.total)
    summary = build_summary(job_dir, expected_total)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text(summary)
        if args.list_infra:
            infra = [t for t in summary["trials"] if t["status"] == "infra_excluded"]
            if infra:
                print("infra trial list:")
                for trial in infra:
                    print(f"- {trial['trial_name']}: {trial['exception_kind']} - {trial['exception_message']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
