#!/usr/bin/env python3
"""Run a resumable, leakage-safe Judge over generic phase templates."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src/SkillDAG_NCF"
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from skilldag.initialize import _http_post_json  # noqa: E402


DATA_ROOT = ROOT / "data/alfworld_task_skill"
DEFAULT_TEMPLATES = DATA_ROOT / "task_skill_v3_phase/judge/templates/templates.jsonl"
DEFAULT_SKILLS = DATA_ROOT / "shared/raw/skills_37.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "task_skill_v3_phase/judge/template_labels"
PROMPT_VERSION = "phase_skill_judge_v3_generic_v1"
VALID_GRADES: set[int | str] = {0, 1, 2, "uncertain"}


SYSTEM_PROMPT = """You are an offline annotator for phase-level skill retrieval in an embodied agent.
Judge each candidate skill for the CURRENT PHASE, while using the full workflow only as context.

Grades:
- 2: directly provides a distinct capability normally required to complete the current phase.
- 1: useful complementary support, prerequisite, verification, or recovery for the current phase, but not directly required.
- 0: irrelevant to the current phase, belongs mainly to another phase, is redundant without adding a distinct capability, or may distract the agent.
- uncertain: the supplied skill text is insufficient or genuinely ambiguous.

Be conservative with grade 2. Similar wording alone is not enough. Do not mark several interchangeable generic skills as required unless each supplies a distinct necessary capability.

Confidence calibration:
- 0.90-1.00 only for an explicit, unambiguous capability match or mismatch.
- 0.70-0.89 for a strong inference.
- below 0.70 when context or scope is ambiguous; use uncertain if needed.

Return exactly one JSON object with no markdown and every skill exactly once:
{"judgments":[{"skill_id":"exact-id","grade":2,"confidence":0.91,"reason":"short reason"}]}
The grade must be 0, 1, 2, or the string "uncertain". Confidence must be in [0,1]."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def body_preview(body: str, max_chars: int = 650) -> str:
    return re.sub(r"\s+", " ", body).strip()[:max_chars]


def prompt_for(template: dict[str, Any], skills: list[dict[str, Any]]) -> str:
    skill_blocks = [
        {
            "skill_id": row["skill_id"],
            "name": row.get("name", ""),
            "description": row.get("description", ""),
            "instructions_preview": body_preview(row.get("skill_body", "")),
        }
        for row in sorted(skills, key=lambda item: item["skill_id"])
    ]
    context = {
        "task_type": template["task_type"],
        "full_workflow": template["generic_full_workflow"],
        "current_phase": template["phase_name"],
        "phase_group": template["phase_group"],
        "current_phase_objective": template["generic_phase_objective"],
        "important": "Judge usefulness for the current phase, not the whole workflow.",
    }
    return "PHASE_CONTEXT:\n" + json.dumps(context, ensure_ascii=False, sort_keys=True) + "\n\nSKILLS:\n" + json.dumps(skill_blocks, ensure_ascii=False, sort_keys=True)


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def parse(content: str, expected_ids: set[str]) -> list[dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response contains no JSON object")
        payload = json.loads(text[start : end + 1])
    judgments = payload.get("judgments") if isinstance(payload, dict) else None
    if not isinstance(judgments, list):
        raise ValueError("response has no judgments array")
    normalized = []
    seen = set()
    for item in judgments:
        skill_id = str(item.get("skill_id", "")).strip()
        grade: int | str = item.get("grade")
        if isinstance(grade, str):
            grade = grade.strip().lower()
            if grade in {"0", "1", "2"}:
                grade = int(grade)
        if skill_id not in expected_ids or skill_id in seen:
            raise ValueError(f"unknown or duplicate skill_id: {skill_id!r}")
        if grade not in VALID_GRADES:
            raise ValueError(f"invalid grade for {skill_id}: {grade!r}")
        confidence = float(item.get("confidence"))
        if not 0 <= confidence <= 1:
            raise ValueError(f"invalid confidence for {skill_id}")
        seen.add(skill_id)
        normalized.append(
            {
                "skill_id": skill_id,
                "grade": grade,
                "confidence": confidence,
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    if seen != expected_ids:
        raise ValueError(f"missing skill IDs: {sorted(expected_ids - seen)}")
    return sorted(normalized, key=lambda row: row["skill_id"])


def cache_name(template_id: str) -> str:
    return hashlib.sha256(template_id.encode()).hexdigest()[:24] + ".json"


def judge_one(template: dict[str, Any], skills: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    prompt = prompt_for(template, skills)
    prompt_hash = "sha256:" + hashlib.sha256((PROMPT_VERSION + "\n" + SYSTEM_PROMPT + "\n" + prompt).encode()).hexdigest()
    cache_path = args.output_dir / "responses" / cache_name(template["template_id"])
    cached_failure = None
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("status") == "success" and cached.get("model") == args.model and cached.get("prompt_hash") == prompt_hash:
            return cached, True
        if cached.get("status") == "failed" and cached.get("model") == args.model and cached.get("prompt_hash") == prompt_hash:
            cached_failure = cached
    request = {
        "model": args.model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8192,
    }
    attempts = list((cached_failure or {}).get("attempts", []))
    result = None
    # A validated failed cache means the full-library response already omitted
    # skills repeatedly. Skip paying for the same malformed request again and
    # move directly to the smaller skill-batch fallback.
    full_attempt_range = range(1, args.max_attempts + 1) if cached_failure is None else ()
    for attempt in full_attempt_range:
        try:
            status, body = _http_post_json(
                args.api_base.rstrip("/") + "/chat/completions",
                {"Authorization": f"Bearer {args.api_key}"},
                request,
                args.timeout,
            )
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {body[:400]}")
            response = json.loads(body)
            try:
                judgments = parse(extract_content(response), {row["skill_id"] for row in skills})
            except Exception as exc:
                attempts.append({"attempt": attempt, "response": response, "parse_error": f"{type(exc).__name__}: {exc}"})
                raise
            attempts.append({"attempt": attempt, "response": response, "parse_error": None})
            result = {
                "status": "success",
                "template_id": template["template_id"],
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash,
                "judgments": judgments,
                "attempts": attempts,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            break
        except Exception as exc:
            if not attempts or attempts[-1].get("attempt") != attempt or "response" not in attempts[-1]:
                attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})

    if result is None and 0 < args.fallback_skill_batch_size < len(skills):
        batch_dir = args.output_dir / "responses" / "skill_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        merged_judgments = []
        batch_records = []
        fallback_failed = False
        for batch_index, start in enumerate(range(0, len(skills), args.fallback_skill_batch_size)):
            skill_batch = skills[start : start + args.fallback_skill_batch_size]
            batch_prompt = prompt_for(template, skill_batch)
            batch_hash = "sha256:" + hashlib.sha256(
                (PROMPT_VERSION + "\n" + SYSTEM_PROMPT + "\n" + batch_prompt).encode()
            ).hexdigest()
            batch_path = batch_dir / f"{cache_path.stem}_batch_{batch_index:02d}.json"
            batch_result = None
            if batch_path.exists():
                candidate = json.loads(batch_path.read_text(encoding="utf-8"))
                if candidate.get("status") == "success" and candidate.get("model") == args.model and candidate.get("prompt_hash") == batch_hash:
                    batch_result = candidate
            if batch_result is None:
                batch_request = {
                    "model": args.model,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": batch_prompt}],
                    "temperature": 0,
                    "max_tokens": 8192,
                }
                batch_attempts = []
                for batch_attempt in range(1, args.max_attempts + 1):
                    try:
                        status, body = _http_post_json(
                            args.api_base.rstrip("/") + "/chat/completions",
                            {"Authorization": f"Bearer {args.api_key}"},
                            batch_request,
                            args.timeout,
                        )
                        if status >= 400:
                            raise RuntimeError(f"HTTP {status}: {body[:400]}")
                        response = json.loads(body)
                        try:
                            judgments = parse(
                                extract_content(response),
                                {row["skill_id"] for row in skill_batch},
                            )
                        except Exception as exc:
                            batch_attempts.append({"attempt": batch_attempt, "response": response, "parse_error": f"{type(exc).__name__}: {exc}"})
                            raise
                        batch_attempts.append({"attempt": batch_attempt, "response": response, "parse_error": None})
                        batch_result = {
                            "status": "success",
                            "model": args.model,
                            "prompt_hash": batch_hash,
                            "judgments": judgments,
                            "attempts": batch_attempts,
                        }
                        break
                    except Exception as exc:
                        if not batch_attempts or batch_attempts[-1].get("attempt") != batch_attempt or "response" not in batch_attempts[-1]:
                            batch_attempts.append({"attempt": batch_attempt, "error": f"{type(exc).__name__}: {exc}"})
                if batch_result is None:
                    batch_result = {"status": "failed", "model": args.model, "prompt_hash": batch_hash, "attempts": batch_attempts}
                batch_path.write_text(json.dumps(batch_result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            batch_records.append({"batch_index": batch_index, **batch_result})
            if batch_result["status"] != "success":
                fallback_failed = True
                break
            merged_judgments.extend(batch_result["judgments"])
        if not fallback_failed:
            merged_judgments.sort(key=lambda row: row["skill_id"])
            if {row["skill_id"] for row in merged_judgments} != {row["skill_id"] for row in skills}:
                raise ValueError("skill-batch fallback did not cover the full skill library")
            result = {
                "status": "success",
                "template_id": template["template_id"],
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash,
                "judgments": merged_judgments,
                "attempts": attempts,
                "skill_batch_fallback": batch_records,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
    if result is None:
        result = {
            "status": "failed",
            "template_id": template["template_id"],
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "attempts": attempts,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates-path", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--skills-path", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--fallback-skill-batch-size", type=int, default=19)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("SKILLDAG_LLM_TIMEOUT_S", "180")))
    parser.add_argument("--model", default=os.environ.get("SKILLDAG_LLM_MODEL", "MiniMax-M2.7"))
    parser.add_argument("--api-base", default=os.environ.get("SKILLDAG_LLM_BASE", "https://yunwu.ai/v1"))
    args = parser.parse_args()
    args.api_key = os.environ.get("SKILLDAG_LLM_API_KEY", "")
    if not args.api_key:
        raise SystemExit("SKILLDAG_LLM_API_KEY is empty")
    templates = load_jsonl(args.templates_path)
    skills = load_jsonl(args.skills_path)
    selected = templates[: args.limit] if args.limit is not None else templates
    results = {}
    cache_hits = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(judge_one, template, skills, args): template["template_id"] for template in selected}
        for index, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            result, cache_hit = future.result()
            results[result["template_id"]] = result
            cache_hits += int(cache_hit)
            print(f"[phase-judge] {index}/{len(selected)} {result['status']} cache={int(cache_hit)} {result['template_id']}", flush=True)

    rows = []
    failures = []
    grades: Counter[str] = Counter()
    confidence_values = []
    usage: Counter[str] = Counter()
    for template in selected:
        result = results[template["template_id"]]
        if result["status"] != "success":
            failures.append(result)
            continue
        response_records = [
            attempt.get("response")
            for attempt in result.get("attempts", [])
            if attempt.get("response")
        ]
        for batch in result.get("skill_batch_fallback", []):
            response_records.extend(
                attempt.get("response")
                for attempt in batch.get("attempts", [])
                if attempt.get("response")
            )
        for response in response_records:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += int((response.get("usage") or {}).get(key, 0) or 0)
        for judgment in result["judgments"]:
            grades[str(judgment["grade"])] += 1
            confidence_values.append(judgment["confidence"])
            rows.append({"template_id": template["template_id"], **judgment, "model": args.model, "prompt_version": PROMPT_VERSION})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "labels.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (args.output_dir / "failures.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in failures), encoding="utf-8")
    report = {
        "schema_version": "skilldag_ncf.v3.phase_template_judge_report.v1",
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "template_count": len(selected),
        "successful_template_count": len(selected) - len(failures),
        "failed_template_count": len(failures),
        "cache_hit_count": cache_hits,
        "new_template_collection_count_this_run": len(selected) - cache_hits,
        "judgment_count": len(rows),
        "grade_counts": dict(sorted(grades.items())),
        "confidence": {
            "min": min(confidence_values) if confidence_values else None,
            "mean": sum(confidence_values) / len(confidence_values) if confidence_values else None,
            "max": max(confidence_values) if confidence_values else None,
            "at_least_0_9": sum(value >= 0.9 for value in confidence_values),
        },
        "validated_response_usage_in_cache": dict(usage),
        "usage_excludes_rejected_or_unparseable_responses": True,
        "uses_expert_plan": False,
        "uses_eval_data": False,
        "outputs": {"labels": "labels.jsonl", "failures": "failures.jsonl", "responses": "responses/"},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
