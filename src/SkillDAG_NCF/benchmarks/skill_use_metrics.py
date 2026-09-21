#!/usr/bin/env python3
"""Skill-use correctness metrics for benchmark outputs.

Gold skills:
  * ALFWorld: derived from each episode oracle high-level plan
    (traj_data.json -> plan.high_pddl[].discrete_action.action).
  * SkillsBench: curated per-task skills in environment/skills, or the
    gold_skills.json emitted by skilldag_benchmark.py for generated tasks.

Used skills:
  * ALFWorld: loaded_skills, relevant_skill_names, and SkillDAG CLI calls.
  * SkillsBench: Harbor result metadata plus agent logs/trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SKILLDAG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALFWORLD_DATA = SKILLDAG_ROOT / "data" / "alfworld" / "data"
DEFAULT_SKILLSBENCH_GOLD_TASKS = SKILLDAG_ROOT / "data" / "tasks" / "tasks"

# Conservative oracle action to skill-id mapping. NoOp is intentionally ignored.
ALFWORLD_ACTION_TO_SKILL: dict[str, str] = {
    "GotoLocation": "alfworld-location-navigator",
    "PickupObject": "alfworld-object-picker",
    "PutObject": "alfworld-object-placer",
    "CleanObject": "alfworld-clean-object",
    "HeatObject": "alfworld-object-heater",
    "CoolObject": "alfworld-object-cooler",
    "ToggleObject": "alfworld-device-operator",
    "SliceObject": "alfworld-tool-user",
}

SKILL_ID_RE = r"[A-Za-z0-9][A-Za-z0-9_.:@+~-]*"
SKILLDAG_SHOW_PATTERNS = [
    re.compile(rf"\bskilldag\s+show\s+({SKILL_ID_RE})\b"),
    re.compile(
        rf"\bskilldag\s+graph\s+"
        rf"(?:show|get-skill|get-dependencies|get-alternatives|get-conflicts|"
        rf"check-set|expand-set|repair-set)\s+({SKILL_ID_RE})\b"
    ),
    re.compile(rf"/(?:opt/skilldag/(?:store|skills)|root/\.[^/\s]+/skills)/({SKILL_ID_RE})(?:/|\b)"),
]


@dataclass
class SkillUseRow:
    task_id: str
    source: str
    gold: set[str]
    used: set[str]
    reward: Any = None

    @property
    def true_positive(self) -> set[str]:
        return self.gold & self.used

    @property
    def false_positive(self) -> set[str]:
        return self.used - self.gold

    @property
    def false_negative(self) -> set[str]:
        return self.gold - self.used

    @property
    def precision(self) -> float:
        return len(self.true_positive) / len(self.used) if self.used else 0.0

    @property
    def recall(self) -> float:
        return len(self.true_positive) / len(self.gold) if self.gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def exact_match(self) -> bool:
        return self.gold == self.used

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "reward": self.reward,
            "gold": sorted(self.gold),
            "used": sorted(self.used),
            "true_positive": sorted(self.true_positive),
            "false_positive": sorted(self.false_positive),
            "false_negative": sorted(self.false_negative),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact_match": self.exact_match,
        }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def default_alfworld_data() -> Path:
    return Path(os.environ.get("ALFWORLD_DATA", str(DEFAULT_ALFWORLD_DATA)))


def _task_name_from_trial_dir(path: Path) -> str:
    return path.name.split("__", 1)[0]


def extract_skill_ids_from_text(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in SKILLDAG_SHOW_PATTERNS:
        for match in pattern.finditer(text):
            skill_id = match.group(1).strip().strip("\"'`")
            if skill_id and "<" not in skill_id and ">" not in skill_id:
                found.add(skill_id)
    return found


def extract_used_skills_from_result(result: dict[str, Any]) -> set[str]:
    used: set[str] = set()

    for key in ("loaded_skills", "relevant_skill_names"):
        value = result.get(key)
        if isinstance(value, list):
            used |= {str(x) for x in value if isinstance(x, str) and x}

    agent_metadata = ((result.get("agent_result") or {}).get("metadata") or {})
    skills_loaded = agent_metadata.get("skills_loaded")
    if isinstance(skills_loaded, list):
        for item in skills_loaded:
            if isinstance(item, str) and item:
                name = Path(item).name if "/" in item else item
                used.add(name[:-3] if name.endswith(".md") else name)
    references_loaded = agent_metadata.get("references_loaded")
    if isinstance(references_loaded, list):
        used |= extract_skill_ids_from_text("\n".join(str(x) for x in references_loaded))

    for record in result.get("cli_invocations") or []:
        if isinstance(record, dict):
            command = record.get("command")
            if isinstance(command, str):
                used |= extract_skill_ids_from_text(command)
    return used


def extract_used_skills_from_trial_dir(trial_dir: Path) -> set[str]:
    used: set[str] = set()
    result_path = trial_dir / "result.json"
    if result_path.exists():
        try:
            result = read_json(result_path)
            if isinstance(result, dict):
                used |= extract_used_skills_from_result(result)
        except Exception:
            pass

    agent_skills_dir = trial_dir / "agent" / "skills"
    if agent_skills_dir.exists():
        used |= {
            p.name
            for p in agent_skills_dir.iterdir()
            if p.is_dir() or p.name.endswith(".md")
        }

    log_candidates = [
        trial_dir / "agent" / "trajectory.json",
        trial_dir / "trajectory.json",
        trial_dir / "trial.log",
        trial_dir / "job.log",
    ]
    log_candidates.extend((trial_dir / "agent").glob("episode-*/*.txt"))
    log_candidates.extend((trial_dir / "agent").glob("command-*/*.txt"))
    log_candidates.extend((trial_dir / "agent").glob("*.txt"))

    for candidate in log_candidates:
        if candidate.exists() and candidate.is_file():
            try:
                used |= extract_skill_ids_from_text(
                    candidate.read_text(encoding="utf-8", errors="ignore")
                )
            except Exception:
                continue
    return used


def alfworld_traj_path(alfworld_data: Path, episode_name: str) -> Path | None:
    root = alfworld_data / "json_2.1.1"
    matches = sorted(root.glob(f"*/{episode_name}/traj_data.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    raise ValueError(f"episode name is not unique under {root}: {episode_name}")


def gold_alfworld_skills(
    traj_path: Path,
    *,
    include_navigation: bool = True,
) -> set[str]:
    data = read_json(traj_path)
    gold: set[str] = set()
    for step in data.get("plan", {}).get("high_pddl", []) or []:
        action = (step.get("discrete_action") or {}).get("action")
        if action == "GotoLocation" and not include_navigation:
            continue
        skill_id = ALFWORLD_ACTION_TO_SKILL.get(action)
        if skill_id:
            gold.add(skill_id)
    return gold


def compute_alfworld_skill_use_row(
    result: dict[str, Any],
    *,
    source: str,
    alfworld_data: Path,
    include_navigation: bool = True,
) -> SkillUseRow:
    episode_name = result.get("name")
    if not isinstance(episode_name, str) or "/" not in episode_name:
        raise ValueError("ALFWorld result is missing a task/trial `name`")
    traj_path = alfworld_traj_path(alfworld_data, episode_name)
    if traj_path is None:
        raise FileNotFoundError(f"cannot locate ALFWorld traj_data.json for {episode_name}")
    return SkillUseRow(
        task_id=episode_name,
        source=source,
        gold=gold_alfworld_skills(traj_path, include_navigation=include_navigation),
        used=extract_used_skills_from_result(result),
        reward=result.get("reward"),
    )


def attach_alfworld_skill_use_metric(
    result: dict[str, Any],
    *,
    alfworld_data: Path | None = None,
    include_navigation: bool = True,
) -> dict[str, Any]:
    row = compute_alfworld_skill_use_row(
        result,
        source=result.get("name", "<in-memory>"),
        alfworld_data=alfworld_data or default_alfworld_data(),
        include_navigation=include_navigation,
    )
    result["skill_use_metric"] = row.as_dict()
    result["skill_use_metric"]["gold_source"] = "alfworld_oracle_high_pddl"
    return result


def iter_alfworld_rows(
    result_dirs: Iterable[Path],
    *,
    alfworld_data: Path,
    include_navigation: bool = True,
) -> list[SkillUseRow]:
    rows: list[SkillUseRow] = []
    for result_dir in result_dirs:
        for result_path in sorted(result_dir.glob("idx_*.json")):
            result = read_json(result_path)
            if isinstance(result, dict):
                rows.append(
                    compute_alfworld_skill_use_row(
                        result,
                        source=str(result_path),
                        alfworld_data=alfworld_data,
                        include_navigation=include_navigation,
                    )
                )
    return rows


def gold_skillsbench_skills(task_dir: Path) -> set[str]:
    generated_gold = task_dir / "environment" / "skilldag" / "gold_skills.json"
    if generated_gold.exists():
        payload = read_json(generated_gold)
        skills = payload.get("gold_skills") if isinstance(payload, dict) else None
        if isinstance(skills, list):
            return {str(x) for x in skills if isinstance(x, str) and x}

    skills_dir = task_dir / "environment" / "skills"
    if not skills_dir.exists():
        return set()
    return {
        p.name
        for p in skills_dir.iterdir()
        if p.is_dir() and (p / "SKILL.md").exists()
    }


def _reward_from_skillsbench_result(result: dict[str, Any]) -> Any:
    rewards = ((result.get("verifier_result") or {}).get("rewards") or {})
    if "reward" in rewards:
        return rewards["reward"]
    return result.get("reward", result.get("score"))


def _gold_dir_for_trial(trial_dir: Path, result: dict[str, Any] | None, gold_tasks_dir: Path) -> Path:
    task_id = _task_name_from_trial_dir(trial_dir)
    if result:
        task_path = (((result.get("config") or {}).get("task") or {}).get("path"))
        if isinstance(task_path, str):
            candidate = Path(task_path)
            if (candidate / "environment" / "skilldag" / "gold_skills.json").exists():
                return candidate
        task_id_value = result.get("task_name")
        if isinstance(task_id_value, str) and task_id_value:
            task_id = task_id_value
    return gold_tasks_dir / task_id


def iter_skillsbench_rows(
    trial_dirs: Iterable[Path],
    *,
    gold_tasks_dir: Path,
) -> list[SkillUseRow]:
    rows: list[SkillUseRow] = []
    for trial_dir in sorted(trial_dirs):
        if not trial_dir.is_dir():
            continue
        result: dict[str, Any] | None = None
        reward = None
        result_path = trial_dir / "result.json"
        if result_path.exists():
            try:
                parsed = read_json(result_path)
                if isinstance(parsed, dict):
                    result = parsed
                    reward = _reward_from_skillsbench_result(parsed)
            except Exception:
                result = None

        task_id = result.get("task_name") if result else _task_name_from_trial_dir(trial_dir)
        if not isinstance(task_id, str) or not task_id:
            task_id = _task_name_from_trial_dir(trial_dir)
        gold_dir = _gold_dir_for_trial(trial_dir, result, gold_tasks_dir)
        gold = gold_skillsbench_skills(gold_dir)
        if not gold:
            continue
        rows.append(
            SkillUseRow(
                task_id=task_id,
                source=str(trial_dir),
                gold=gold,
                used=extract_used_skills_from_trial_dir(trial_dir),
                reward=reward,
            )
        )
    return rows


def collect_skillsbench_trial_dirs(paths: Iterable[Path]) -> list[Path]:
    trial_dirs: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        result_path = path / "result.json"
        if result_path.exists():
            try:
                payload = read_json(result_path)
                if isinstance(payload, dict) and payload.get("task_name"):
                    trial_dirs.append(path)
                    continue
            except Exception:
                pass
        trial_dirs.extend(
            child
            for child in sorted(path.iterdir())
            if child.is_dir() and (child / "result.json").exists()
        )
    return trial_dirs


def aggregate_rows(rows: list[SkillUseRow]) -> dict[str, Any]:
    tp = sum(len(row.true_positive) for row in rows)
    fp = sum(len(row.false_positive) for row in rows)
    fn = sum(len(row.false_negative) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_count = sum(1 for row in rows if row.exact_match)
    return {
        "n_tasks": len(rows),
        "micro": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "macro": {
            "precision": statistics.fmean([row.precision for row in rows]) if rows else 0.0,
            "recall": statistics.fmean([row.recall for row in rows]) if rows else 0.0,
            "f1": statistics.fmean([row.f1 for row in rows]) if rows else 0.0,
        },
        "exact_match": {
            "count": exact_count,
            "rate": exact_count / len(rows) if rows else 0.0,
        },
        "mean_gold_skills": statistics.fmean([len(row.gold) for row in rows]) if rows else 0.0,
        "mean_used_skills": statistics.fmean([len(row.used) for row in rows]) if rows else 0.0,
    }


def build_payload(rows: list[SkillUseRow], *, gold_source: str) -> dict[str, Any]:
    return {
        "metric": "skill_use_correctness",
        "definition": {
            "precision": "|used_skills intersect gold_skills| / |used_skills|",
            "recall": "|used_skills intersect gold_skills| / |gold_skills|",
            "f1": "harmonic mean of precision and recall",
        },
        "gold_source": gold_source,
        "summary": aggregate_rows(rows),
        "per_task": [row.as_dict() for row in rows],
    }


def print_summary(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    micro = summary["micro"]
    macro = summary["macro"]
    exact = summary["exact_match"]
    print(f"tasks: {summary['n_tasks']}")
    print(
        "micro: "
        f"P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f} "
        f"(tp={micro['tp']} fp={micro['fp']} fn={micro['fn']})"
    )
    print(
        "macro: "
        f"P={macro['precision']:.4f} R={macro['recall']:.4f} F1={macro['f1']:.4f}"
    )
    print(
        f"exact_match: {exact['count']}/{summary['n_tasks']} "
        f"({exact['rate']:.4f})"
    )
    print(
        f"mean skills: gold={summary['mean_gold_skills']:.2f} "
        f"used={summary['mean_used_skills']:.2f}"
    )


def write_payload(payload: dict[str, Any], output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")


def write_alfworld_skill_use_metrics(
    result_dirs: Iterable[Path],
    *,
    output: Path | None = None,
    alfworld_data: Path | None = None,
    include_navigation: bool = True,
) -> dict[str, Any]:
    rows = iter_alfworld_rows(
        result_dirs,
        alfworld_data=alfworld_data or default_alfworld_data(),
        include_navigation=include_navigation,
    )
    payload = build_payload(rows, gold_source="alfworld_oracle_high_pddl")
    write_payload(payload, output)
    return payload


def write_skillsbench_skill_use_metrics(
    trial_roots: Iterable[Path],
    *,
    output: Path | None = None,
    gold_tasks_dir: Path | None = None,
) -> dict[str, Any]:
    trial_dirs = collect_skillsbench_trial_dirs(trial_roots)
    rows = iter_skillsbench_rows(
        trial_dirs,
        gold_tasks_dir=gold_tasks_dir or DEFAULT_SKILLSBENCH_GOLD_TASKS,
    )
    payload = build_payload(rows, gold_source="skillsbench_environment_skills")
    write_payload(payload, output)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skill_use_metrics")
    sub = parser.add_subparsers(dest="command", required=True)

    alf = sub.add_parser("alfworld", help="Compute ALFWorld skill-use correctness")
    alf.add_argument("--results-dir", type=Path, nargs="+", required=True)
    alf.add_argument("--alfworld-data", type=Path, default=DEFAULT_ALFWORLD_DATA)
    alf.add_argument("--exclude-navigation", action="store_true")
    alf.add_argument("--output", type=Path)
    alf.add_argument("--json", action="store_true", help="Print full JSON payload")

    sb = sub.add_parser("skillsbench", help="Compute SkillsBench skill-use correctness")
    sb.add_argument("--trial-dir", type=Path, nargs="+", required=True)
    sb.add_argument("--gold-tasks-dir", type=Path, default=DEFAULT_SKILLSBENCH_GOLD_TASKS)
    sb.add_argument("--output", type=Path)
    sb.add_argument("--json", action="store_true", help="Print full JSON payload")

    args = parser.parse_args(argv)
    if args.command == "alfworld":
        payload = write_alfworld_skill_use_metrics(
            args.results_dir,
            output=args.output,
            alfworld_data=args.alfworld_data,
            include_navigation=not args.exclude_navigation,
        )
    elif args.command == "skillsbench":
        payload = write_skillsbench_skill_use_metrics(
            args.trial_dir,
            output=args.output,
            gold_tasks_dir=args.gold_tasks_dir,
        )
    else:
        return 2

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_summary(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
