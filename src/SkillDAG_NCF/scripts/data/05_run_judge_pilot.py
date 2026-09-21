#!/usr/bin/env python3
"""Run MiniMax-M2.7 Judge over a selected task collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from skilldag.data_pipeline import run_judge_pilot
from skilldag.initialize import _http_post_json


def _defaults() -> tuple[Path, Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[4]
    data = root / ".runtime" / "skilldag_ncf" / "data"
    return (
        data / "raw" / "tasks.jsonl",
        data / "raw" / "skills.jsonl",
        data / "intermediate" / "pilot_combined_candidates.jsonl",
        data / "reports" / "task_semantic_pilot_selection.json",
        data,
    )


def main() -> int:
    tasks, skills, candidates, selection, output = _defaults()
    parser = argparse.ArgumentParser(description="Collect resumable Judge weak-label evidence.")
    parser.add_argument("--tasks-path", type=Path, default=tasks)
    parser.add_argument("--skills-path", type=Path, default=skills)
    parser.add_argument("--candidates-path", type=Path, default=candidates)
    parser.add_argument("--selection-path", type=Path, default=selection)
    parser.add_argument("--output-root", type=Path, default=output)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--collection-name", default="pilot")
    parser.add_argument("--fallback-cache-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    api_key = os.environ.get("SKILLDAG_LLM_API_KEY", "")
    api_base = os.environ.get("SKILLDAG_LLM_BASE", "https://yunwu.ai/v1").rstrip("/")
    model = os.environ.get("SKILLDAG_LLM_MODEL", "MiniMax-M2.7")
    timeout = int(os.environ.get("SKILLDAG_LLM_TIMEOUT_S", "180"))
    if not api_key:
        raise SystemExit("SKILLDAG_LLM_API_KEY is empty; use the companion .sh script.")

    def chat(payload):
        status, body = _http_post_json(
            f"{api_base}/chat/completions",
            {"Authorization": f"Bearer {api_key}"},
            payload,
            timeout,
        )
        if status >= 400:
            raise RuntimeError(f"chat completion HTTP {status}: {body[:300]}")
        return json.loads(body)

    def show_progress(done, total, record_id, status):
        print(f"[judge] {done}/{total} {status} {record_id}", flush=True)

    report = run_judge_pilot(
        args.tasks_path,
        args.skills_path,
        args.candidates_path,
        args.selection_path,
        args.output_root,
        model,
        chat,
        max_workers=args.max_workers,
        max_attempts=args.max_attempts,
        limit=args.limit,
        progress=show_progress,
        collection_name=args.collection_name,
        fallback_cache_dirs=args.fallback_cache_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["failed_task_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
