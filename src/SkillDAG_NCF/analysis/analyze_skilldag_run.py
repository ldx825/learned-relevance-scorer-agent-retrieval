#!/usr/bin/env python3
"""Analyze SkillDAG SkillsBench job outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics as stats
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def load_score_module():
    path = Path(__file__).with_name("score_skillsbench_gos.py")
    spec = importlib.util.spec_from_file_location("score_skillsbench_gos", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCORE = load_score_module()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def duration_seconds(block: dict[str, Any] | None) -> float | None:
    if not isinstance(block, dict):
        return None
    start = parse_dt(block.get("started_at"))
    end = parse_dt(block.get("finished_at"))
    if not start or not end:
        return None
    return max((end - start).total_seconds(), 0.0)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "median", "p90", "p95")}
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = (len(ordered) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(ordered) - 1)
        frac = idx - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return {
        "mean": stats.fmean(ordered),
        "median": stats.median(ordered),
        "p90": pct(0.90),
        "p95": pct(0.95),
    }


def trial_dirs(job_dir: Path) -> list[Path]:
    return [p for p in sorted(job_dir.iterdir()) if SCORE.is_trial_dir(p)]


def base_task(trial_dir: Path) -> str:
    return SCORE.base_task_name(trial_dir)


def result_for(trial_dir: Path) -> dict[str, Any] | None:
    path = trial_dir / "result.json"
    if not path.exists():
        return None
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return None


def latency_metrics(job_dir: Path) -> dict[str, Any]:
    latencies: list[float] = []
    per_task: dict[str, list[float]] = {}
    calls_per_trial: dict[str, int] = {}
    agent_times: list[float] = []
    verifier_times: list[float] = []
    timeout_counts: Counter[str] = Counter()

    for trial in trial_dirs(job_dir):
        task = base_task(trial)
        calls = 0
        for prompt in sorted((trial / "agent").glob("episode-*/prompt.txt")):
            response = prompt.with_name("response.txt")
            if not response.exists():
                continue
            latency = response.stat().st_mtime - prompt.stat().st_mtime
            if latency >= 0:
                latencies.append(latency)
                per_task.setdefault(task, []).append(latency)
                calls += 1
        calls_per_trial[trial.name] = calls

        result = result_for(trial)
        if result:
            agent = duration_seconds(result.get("agent_execution"))
            verifier = duration_seconds(result.get("verifier"))
            if agent is not None:
                agent_times.append(agent)
            if verifier is not None:
                verifier_times.append(verifier)
            exc = (result.get("exception_info") or {}).get("exception_type")
            if exc and "Timeout" in exc:
                timeout_counts[str(exc)] += 1

    return {
        "llm_response_latency_seconds": {"n": len(latencies), **quantiles(latencies)},
        "per_task_mean_latency_seconds": {k: stats.fmean(v) for k, v in sorted(per_task.items()) if v},
        "calls_per_trial": {"n_trials": len(calls_per_trial), **quantiles(list(calls_per_trial.values()))},
        "agent_execution_seconds": {"n": len(agent_times), **quantiles(agent_times)},
        "verifier_seconds": {"n": len(verifier_times), **quantiles(verifier_times)},
        "timeout_counts": dict(sorted(timeout_counts.items())),
    }


def json_objects(text: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def trajectory(trial: Path) -> list[dict[str, Any]]:
    path = trial / "agent" / "trajectory.json"
    if not path.exists():
        return []
    try:
        data = load_json(path)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def command_texts(items: list[dict[str, Any]]) -> list[str]:
    texts = []
    for item in items:
        response = item.get("response")
        if isinstance(response, str):
            texts.append(response)
    return texts


def first_search_top5(items: list[dict[str, Any]]) -> list[str]:
    for item in items:
        prompt = item.get("prompt")
        if not isinstance(prompt, str):
            continue
        for obj in json_objects(prompt):
            if obj.get("ok") is True and obj.get("op") == "search":
                result = obj.get("result") or {}
                return [str(m.get("skill_id")) for m in result.get("matches") or [] if m.get("skill_id")][:5]
    return []


def extract_skill_ids(texts: list[str], pattern: str) -> set[str]:
    ids: set[str] = set()
    for text in texts:
        for match in re.finditer(pattern, text):
            for token in re.split(r"\s+", match.group(1).strip()):
                token = token.strip("'\"` ,")
                if token and not token.startswith("-") and token not in {"\\n", "depends_on", "composes_with", "similar_to"}:
                    ids.add(token)
    return ids


def gold_for(gold_root: Path, task: str) -> set[str]:
    path = gold_root / task / "environment" / "skilldag" / "gold_skills.json"
    if not path.exists():
        return set()
    data = load_json(path)
    return {str(skill) for skill in data.get("gold_skills") or []}


def hit_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["gold"] and row["retrieved_top5"]]
    hit5 = sum(row["hit_at_5"] for row in valid)
    hit1 = sum(row["hit_at_1"] for row in valid)
    mrr = sum(row["mrr"] for row in valid)
    denom = len(valid) or 1
    return {"n_tasks": len(valid), "hit_at_5": hit5 / denom, "top1": hit1 / denom, "mrr": mrr / denom}


def retrieval_metrics(job_dir: Path, gold_root: Path) -> dict[str, Any]:
    rows = []
    command_counts: Counter[str] = Counter()
    tasks_with_gold = 0
    for trial in trial_dirs(job_dir):
        task = base_task(trial)
        gold = gold_for(gold_root, task)
        tasks_with_gold += bool(gold)
        items = trajectory(trial)
        texts = command_texts(items)
        retrieved = first_search_top5(items)
        shown = extract_skill_ids(texts, r"skilldag\s+show\s+([^\\n\"]+)")
        used = set(shown)
        used |= extract_skill_ids(texts, r"skilldag\s+graph\s+get-(?:skill|dependencies|alternatives|conflicts)\s+([^\\n\"]+)")
        for text in texts:
            command_counts["search"] += len(re.findall(r"skilldag\s+graph\s+search", text))
            command_counts["show"] += len(re.findall(r"skilldag\s+show\s+", text))
            command_counts["get_neighbor"] += len(re.findall(r"skilldag\s+graph\s+get-(?:dependencies|alternatives|conflicts)", text))

        first_rank = next((i for i, skill in enumerate(retrieved, 1) if skill in gold), None)
        rows.append({
            "task_id": task,
            "gold": sorted(gold),
            "retrieved_top5": retrieved,
            "hit_at_5": first_rank is not None,
            "hit_at_1": bool(retrieved and retrieved[0] in gold),
            "first_hit_rank": first_rank,
            "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
            "shown_skills": sorted(shown),
            "used_skill_ids": sorted(used),
            "show_hit": bool(gold & shown),
            "use_hit": bool(gold & used),
            "has_show": bool(shown),
            "has_use": bool(used),
        })

    denom = len(rows) or 1
    return {
        "definition": "fixed top-k uses first observed SkillDAG graph search matches; show/use use explicit skill IDs in trajectory commands",
        "gold_root": str(gold_root),
        "gold_coverage": tasks_with_gold / denom,
        "fixed_top_k": hit_metrics(rows),
        "show_hit": sum(r["show_hit"] for r in rows) / denom,
        "show_coverage": sum(r["has_show"] for r in rows) / denom,
        "use_hit": sum(r["use_hit"] for r in rows) / denom,
        "use_coverage": sum(r["has_use"] for r in rows) / denom,
        "command_counts": dict(sorted(command_counts.items())),
        "rows": rows,
    }


def analyze(job_dir: Path, total: int | None, gold_root: Path | None) -> dict[str, Any]:
    expected, _attempts = SCORE.expected_trial_count(job_dir, total)
    out = {
        "job_dir": str(job_dir),
        "score": SCORE.build_summary(job_dir, expected),
        "latency": latency_metrics(job_dir),
    }
    if gold_root:
        out["retrieval"] = retrieval_metrics(job_dir, gold_root)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--job-name")
    parser.add_argument("--total", type=int, default=87)
    parser.add_argument("--gold-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    job_dir = SCORE.resolve_job_dir(args.path, args.job_name)
    data = analyze(job_dir, args.total, args.gold_root)
    if args.output:
        write_json(args.output, data)
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
