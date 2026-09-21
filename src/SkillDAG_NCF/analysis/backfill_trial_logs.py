#!/usr/bin/env python3
"""Backfill Harbor trial.log files from existing trial artifacts.

Harbor creates trial.log, but most SkillsBench agent details are written to
agent/trajectory.json instead. This helper is intentionally read-only with
respect to experiment semantics: it only writes trial.log for inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[... truncated {omitted} chars ...]"


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"_error": f"failed to parse {path.name}: {exc}"}


def _extract_reward(result: dict[str, Any]) -> str:
    verifier_result = result.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    if not rewards:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in rewards.items())


def _extract_exception(result: dict[str, Any]) -> str:
    exc = result.get("exception_info")
    if not exc:
        return "-"
    exc_type = exc.get("type") or exc.get("exception_type") or "exception"
    message = exc.get("message") or exc.get("exception_message") or ""
    return f"{exc_type}: {message}".strip()


def _format_trial_log(trial_dir: Path, max_field_chars: int) -> str:
    lines: list[str] = []
    lines.append(f"trial: {trial_dir.name}")
    lines.append(f"path: {trial_dir}")

    result = _read_json(trial_dir / "result.json")
    if isinstance(result, dict):
        lines.extend(
            [
                "",
                "== Result ==",
                f"task: {result.get('task_name', '-')}",
                f"started_at: {result.get('started_at', '-')}",
                f"finished_at: {result.get('finished_at', '-')}",
                f"reward: {_extract_reward(result)}",
                f"exception: {_extract_exception(result)}",
            ]
        )
    elif result is None:
        lines.extend(["", "== Result ==", "pending: result.json not written yet"])

    trajectory = _read_json(trial_dir / "agent" / "trajectory.json")
    if isinstance(trajectory, list):
        lines.extend(["", "== Agent Trajectory =="])
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            episode = step.get("episode", "?")
            prompt = _shorten(str(step.get("prompt", "")), max_field_chars)
            response = _shorten(str(step.get("response", "")), max_field_chars)
            lines.extend(
                [
                    "",
                    f"-- episode {episode} prompt --",
                    prompt,
                    "",
                    f"-- episode {episode} response --",
                    response,
                ]
            )
    elif isinstance(trajectory, dict):
        lines.extend(
            [
                "",
                "== Agent Trajectory ==",
                json.dumps(trajectory, indent=2, ensure_ascii=False),
            ]
        )
    else:
        lines.extend(["", "== Agent Trajectory ==", "pending: trajectory.json not written yet"])

    verifier_dir = trial_dir / "verifier"
    verifier_files = [
        verifier_dir / "reward.txt",
        verifier_dir / "reward.json",
        verifier_dir / "test-stdout.txt",
        verifier_dir / "test-stderr.txt",
    ]
    existing_verifier_files = [path for path in verifier_files if path.exists()]
    if existing_verifier_files:
        lines.extend(["", "== Verifier =="])
        for path in existing_verifier_files:
            content = _shorten(path.read_text(encoding="utf-8", errors="replace"), max_field_chars)
            lines.extend(["", f"-- {path.name} --", content])

    exception_path = trial_dir / "exception.txt"
    if exception_path.exists():
        lines.extend(
            [
                "",
                "== Exception Traceback ==",
                _shorten(exception_path.read_text(encoding="utf-8", errors="replace"), max_field_chars),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def _iter_trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in job_dir.iterdir()
        if path.is_dir() and (path / "config.json").exists()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path, help="Harbor job directory")
    parser.add_argument(
        "--max-field-chars",
        type=int,
        default=20000,
        help="Maximum prompt/response/verifier chars per section",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite non-empty trial.log files too",
    )
    args = parser.parse_args()

    job_dir = args.job_dir.expanduser().resolve()
    if not job_dir.is_dir():
        raise SystemExit(f"not a directory: {job_dir}")

    written = 0
    skipped = 0
    for trial_dir in _iter_trial_dirs(job_dir):
        log_path = trial_dir / "trial.log"
        if log_path.exists() and log_path.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            continue
        log_path.write_text(
            _format_trial_log(trial_dir, args.max_field_chars),
            encoding="utf-8",
        )
        written += 1

    print(f"job_dir={job_dir}")
    print(f"written={written}")
    print(f"skipped_nonempty={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
