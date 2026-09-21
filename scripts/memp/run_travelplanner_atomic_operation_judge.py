#!/usr/bin/env python3
"""Resumable GPT-4o Judge for atomic task-subgoal--operation candidates."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src/SkillDAG_NCF"
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))
from skilldag.initialize import _http_post_json  # noqa: E402

PROMPT_VERSION = "travelplanner_atomic_operation_judge_v1"
VALID_GRADES: set[int | str] = {0, 1, 2, "uncertain"}
SYSTEM_PROMPT = """You judge task-subgoal to procedural-operation utility for TravelPlanner.

Use only the visible full task, current subgoal, parent Memory source query, and grounded operation text.
- Grade 2: the operation directly and correctly fulfills the current subgoal, including any explicit required value.
- Grade 1: it is a distinct necessary complement, prerequisite, evidence, verification, or repair step, but does not fulfill the subgoal alone.
- Grade 0: it is irrelevant, redundant, or conflicts with an explicit task value.
- uncertain: the visible evidence cannot establish usefulness or harm.

Important:
- Judge the atomic operation, not whether the whole parent Memory solves the complete task.
- A source-task value mismatch is not automatically harmful when the operation is explicitly value-agnostic and transferable.
- no-flight and no-self-driving are different; room types and house rules must not be silently exchanged.
- If the operation goal/procedure/success evidence explicitly names a cuisine, house rule, room type, or transportation value that differs from the task's required value, it MUST be Grade 0. Generic words such as "requested cuisines" do not override a concrete conflicting list elsewhere in the operation.
- Notebook recording is Grade 1 only when the operation records evidence specifically needed by the current subgoal. Recording transportation is Grade 0 for cuisine, room-type, or house-rule subgoals.
- Prefer uncertain over inventing an unsupported transfer.
- Keep at most one Grade-2 anchor in this compact candidate group.

Return exactly one JSON object with every operation_id exactly once and no markdown:
{"judgments":[{"operation_id":"memory_000:route_search:00","grade":2,"confidence":0.88,"evidence":"short exact evidence","reason":"short reason"}]}
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def content(response: dict[str, Any]) -> str:
    return str((((response.get("choices") or [{}])[0].get("message") or {}).get("content", "")))


def parse(text: str, expected: set[str], max_grade2: int = 1) -> list[dict[str, Any]]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        payload = json.loads(value[value.find("{"): value.rfind("}") + 1])
    judgments = payload.get("judgments") if isinstance(payload, dict) else None
    if not isinstance(judgments, list):
        raise ValueError("missing judgments")
    rows, seen = [], set()
    for raw in judgments:
        operation_id = str(raw.get("operation_id", ""))
        grade: int | str = raw.get("grade")
        if isinstance(grade, str) and grade in {"0", "1", "2"}:
            grade = int(grade)
        confidence = float(raw.get("confidence"))
        evidence, reason = str(raw.get("evidence", "")).strip(), str(raw.get("reason", "")).strip()
        if operation_id not in expected or operation_id in seen or grade not in VALID_GRADES:
            raise ValueError("invalid operation ID or grade")
        if not 0 <= confidence <= 1 or not evidence or not reason:
            raise ValueError("invalid confidence/evidence/reason")
        seen.add(operation_id)
        rows.append({"operation_id": operation_id, "grade": grade, "confidence": confidence, "evidence": evidence, "reason": reason})
    if seen != expected:
        raise ValueError(f"missing operation IDs: {sorted(expected - seen)}")
    if sum(row["grade"] == 2 for row in rows) > max_grade2:
        raise ValueError(f"more than {max_grade2} Grade-2 anchors")
    return rows


def judge(batch: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    prompt = json.dumps({key: value for key, value in batch.items() if key not in {"task_id", "seed_fields_visible_to_judge", "validation_or_test_used"}}, ensure_ascii=False, sort_keys=True)
    prompt_hash = "sha256:" + hashlib.sha256((args.prompt_version + args.system_prompt + prompt).encode()).hexdigest()
    cache = args.output_dir / "responses" / f"group_{int(batch['task_id']):04d}.json"
    if cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
        if result.get("status") == "success" and result.get("prompt_hash") == prompt_hash:
            return result, True
    if args.request_delay > 0:
        time.sleep(args.request_delay)
    attempts, result = [], None
    for attempt in range(1, args.max_attempts + 1):
        try:
            request = {"model": args.model, "messages": [{"role": "system", "content": args.system_prompt}, {"role": "user", "content": prompt}], "temperature": 0, "max_tokens": args.max_tokens}
            status, body = _http_post_json(args.api_base.rstrip("/") + "/chat/completions", {"Authorization": f"Bearer {args.api_key}"}, request, args.timeout)
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {body[:500]}")
            response = json.loads(body)
            rows = parse(
                content(response),
                {row["operation_id"] for row in batch["candidates"]},
                max_grade2=args.max_grade2,
            )
            attempts.append({"attempt": attempt, "response": response, "error": None})
            result = {"status": "success", "task_id": batch["task_id"], "subgoal_id": batch["subgoal_id"], "prompt_hash": prompt_hash, "judgments": rows, "attempts": attempts}
            break
        except Exception as exc:
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
            if attempt < args.max_attempts and args.retry_backoff > 0:
                time.sleep(args.retry_backoff * attempt)
    if result is None:
        result = {"status": "failed", "task_id": batch["task_id"], "subgoal_id": batch["subgoal_id"], "prompt_hash": prompt_hash, "attempts": attempts}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api-base", default=os.environ.get("SKILLDAG_LLM_BASE", "https://yunwu.ai/v1"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to wait before each uncached group; useful for rate-limited relays.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=0.0,
        help="Linear backoff base in seconds between failed attempts.",
    )
    parser.add_argument(
        "--max-grade2",
        type=int,
        default=1,
        help="Maximum equivalent direct endpoints allowed in one same-role group.",
    )
    parser.add_argument(
        "--same-role-group",
        action="store_true",
        help="Judge parent-Memory endpoints of one already-fixed operation role.",
    )
    parser.add_argument(
        "--runtime-rebound-group",
        action="store_true",
        help="Treat current applicability values as authoritative, matching online rendering.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_grade2 < 1:
        raise SystemExit("--max-grade2 must be positive")
    if args.request_delay < 0 or args.retry_backoff < 0:
        raise SystemExit("request delay and retry backoff must be non-negative")
    args.prompt_version = f"{PROMPT_VERSION}.max_grade2_{args.max_grade2}"
    args.system_prompt = SYSTEM_PROMPT
    if args.max_grade2 > 1:
        args.system_prompt = SYSTEM_PROMPT.replace(
            "- Keep at most one Grade-2 anchor in this compact candidate group.",
            "- Multiple Grade-2 endpoints are allowed only when each independently and equally "
            "fulfills the same required operation. Do not demote an otherwise equivalent correct "
            "endpoint merely to force a unique winner.",
        )
    if args.same_role_group:
        args.prompt_version += ".same_role_v1"
        args.system_prompt += """

This request compares parent-Memory endpoints for one already-fixed required_operation_key.
- Judge whether each candidate faithfully performs that required operation role, not whether it completes the entire planning phase alone.
- Do not penalize an operation for a companion role handled elsewhere. For example, dining_search may be Grade 2 when it correctly searches and records restaurants even though cuisine_coverage is a separate operation; route_search need not independently perform route_closure; lodging_search need not independently perform final budget audit.
- A generic transferable reference to the task's requested constraint is acceptable. A concrete source value that conflicts with the visible task remains Grade 0.
- Compare procedural completeness and grounded evidence among endpoints of the same role.
"""
    if args.runtime_rebound_group:
        args.prompt_version += ".runtime_rebound_v1"
        args.system_prompt += """

These candidates reproduce the online runtime-rebinding interface.
- Current applicability constraint is authoritative and comes only from the visible current task.
- Parent Memory source query and reusable source evidence are provenance, not current-task requirements.
- Do not penalize a candidate merely because its provenance mentions a different source-task value when the runtime operation explicitly says to ignore source constraints and shows the correct current applicability value.
- Grade 0 is still required when the runtime procedure itself retains or enforces a value that conflicts with the current task.
"""
    batches = read_jsonl(args.candidates)
    if args.dry_run:
        print(json.dumps({"group_count": len(batches), "pair_count": sum(len(row["candidates"]) for row in batches), "prompt_chars": sum(len(args.system_prompt) + len(json.dumps(row, ensure_ascii=False)) for row in batches), "validation_or_test_used": False}, indent=2))
        return 0
    args.api_key = os.environ.get("SKILLDAG_LLM_API_KEY", "")
    if not args.api_key:
        raise SystemExit("SKILLDAG_LLM_API_KEY is empty")
    results, hits = {}, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(judge, row, args): row["task_id"] for row in batches}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result, hit = future.result(); results[result["task_id"]] = result; hits += int(hit)
            print(f"[atomic-judge] {index}/{len(batches)} {result['status']} cache={int(hit)}", flush=True)
    labels, failures, usage = [], [], Counter()
    for batch in batches:
        result = results[batch["task_id"]]
        if result["status"] != "success": failures.append(result); continue
        for attempt in result["attempts"]:
            response = attempt.get("response") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += int((response.get("usage") or {}).get(key, 0) or 0)
        for row in result["judgments"]:
            labels.append({"task_id": batch["task_id"], "subgoal_id": batch["subgoal_id"], **row, "model": args.model, "prompt_version": args.prompt_version, "validation_or_test_used": False})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "labels.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
    (args.output_dir / "failures.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in failures), encoding="utf-8")
    report = {"group_count": len(batches), "successful_group_count": len(batches) - len(failures), "failed_group_count": len(failures), "judgment_count": len(labels), "grade_counts": dict(Counter(str(row["grade"]) for row in labels)), "cache_hit_count": hits, "usage": dict(usage), "validation_or_test_used": False}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
