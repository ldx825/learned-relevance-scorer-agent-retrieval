#!/usr/bin/env python3
"""Summarize and pair the four paper-main ALFWorld experiment arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_run(path: Path) -> dict[str, dict]:
    rows = {}
    for file_path in sorted(path.glob("idx_*.json")):
        row = json.loads(file_path.read_text(encoding="utf-8"))
        task_id = str(row.get("name") or row.get("query") or file_path.stem)
        rows[task_id] = row
    return rows


def token_total(row: dict) -> int:
    usage = row.get("token_usage") or {}
    if "total_tokens" in usage:
        return int(usage.get("total_tokens") or 0)
    if "prompt" in usage or "completion" in usage:
        return int(usage.get("prompt") or 0) + int(usage.get("completion") or 0)
    return 0


def summary(rows: dict[str, dict]) -> dict:
    values = list(rows.values())
    return {
        "tasks": len(values),
        "successes": sum(bool(row.get("reward")) for row in values),
        "success_rate_percent": (
            100.0 * sum(bool(row.get("reward")) for row in values) / len(values)
            if values
            else 0.0
        ),
        "mean_env_steps": mean(float(row.get("steps") or 0) for row in values)
        if values
        else 0.0,
        "mean_agent_tokens": mean(token_total(row) for row in values)
        if values
        else 0.0,
    }


def paired(base: dict[str, dict], ncf: dict[str, dict]) -> dict:
    common = sorted(set(base) & set(ncf))
    base_only = sorted(set(base) - set(ncf))
    ncf_only = sorted(set(ncf) - set(base))
    ncf_wins = [
        task for task in common
        if bool(ncf[task].get("reward")) and not bool(base[task].get("reward"))
    ]
    base_wins = [
        task for task in common
        if bool(base[task].get("reward")) and not bool(ncf[task].get("reward"))
    ]
    return {
        "common_tasks": len(common),
        "base_only_tasks": base_only,
        "ncf_only_tasks": ncf_only,
        "ncf_wins": ncf_wins,
        "base_wins": base_wins,
        "net_success_gain": len(ncf_wins) - len(base_wins),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("gos", "gos_ncf", "skilldag", "skilldag_ncf"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = {
        "gos": load_run(args.gos),
        "gos_ncf": load_run(args.gos_ncf),
        "skilldag": load_run(args.skilldag),
        "skilldag_ncf": load_run(args.skilldag_ncf),
    }
    report = {
        "arms": {name: summary(rows) for name, rows in runs.items()},
        "paired": {
            "gos": paired(runs["gos"], runs["gos_ncf"]),
            "skilldag": paired(runs["skilldag"], runs["skilldag_ncf"]),
        },
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
