"""Resumable LLM-as-a-Judge weak-label collection."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .action_candidates import _atomic_json, _atomic_jsonl, _load_jsonl, _sha256_text


PROMPT_VERSION = "task_skill_judge_v1"
VALID_NECESSITY = {"required", "helpful", "irrelevant", "harmful", "uncertain"}
ChatFunction = Callable[[dict[str, Any]], dict[str, Any]]
ProgressFunction = Callable[[int, int, str, str], None]


SYSTEM_PROMPT = """You are an offline data annotator for ALFWorld task-skill retrieval.
Judge whether each candidate skill would be useful for completing the given task.
The expert plan is privileged training-time evidence. Do not assume a skill is useful merely because its wording is similar.

Labels:
- required: the capability is necessary for this task, and no listed alternative makes it redundant.
- helpful: useful guidance or a valid alternative, but not strictly necessary.
- irrelevant: not useful for this task.
- harmful: following it would likely cause an incorrect action or state.
- uncertain: evidence is insufficient or contradictory.

Return exactly one JSON object and no markdown. Include every candidate skill exactly once:
{"judgments":[{"skill_id":"exact-id","necessity":"required|helpful|irrelevant|harmful|uncertain","relevance":0.0,"confidence":0.0,"reason":"one short sentence"}]}
Both relevance and confidence must be numbers from 0 to 1."""


def _body_preview(body: str, max_chars: int = 1000) -> str:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    text = re.sub(r"\s+", " ", " ".join(lines))
    return text[:max_chars]


def _build_prompt(task: dict[str, Any], candidates: list[dict[str, Any]], skills: dict[str, dict[str, Any]]) -> str:
    plan = [
        {"action": step.get("action", ""), "args": step.get("args", [])}
        for step in task.get("plan_high_pddl", [])
        if step.get("action") and step.get("action") != "NoOp"
    ]
    candidate_blocks = []
    for row in candidates:
        skill = skills[row["skill_id"]]
        candidate_blocks.append(
            {
                "skill_id": row["skill_id"],
                "description": skill.get("description", ""),
                "body_preview": _body_preview(skill.get("skill_body", "")),
                "retrieval_evidence": {
                    "sources": row.get("candidate_sources", []),
                    "task_rank": row.get("task_semantic_rank"),
                    "task_cosine": row.get("task_cosine_score"),
                    "action_rank": row.get("action_semantic_rank"),
                    "matched_actions": row.get("matched_actions", []),
                },
            }
        )
    return (
        "TASK:\n"
        + json.dumps(
            {
                "task_id": task["record_id"],
                "task_type": task.get("task_type", ""),
                "task_text": task.get("task_text", ""),
                "expert_high_level_plan": plan,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n\nCANDIDATE_SKILLS:\n"
        + json.dumps(candidate_blocks, ensure_ascii=False, sort_keys=True)
    )


def _extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("chat response has no choices")
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _parse_judgments(content: str, expected_skill_ids: set[str]) -> list[dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge response does not contain a JSON object")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict) or not isinstance(parsed.get("judgments"), list):
        raise ValueError("Judge response must contain a judgments array")
    judgments = parsed["judgments"]
    seen: set[str] = set()
    normalized = []
    for item in judgments:
        if not isinstance(item, dict):
            raise ValueError("each judgment must be an object")
        skill_id = str(item.get("skill_id", "")).strip()
        necessity = str(item.get("necessity", "")).strip().lower()
        if skill_id not in expected_skill_ids:
            raise ValueError(f"unknown candidate skill_id: {skill_id!r}")
        if skill_id in seen:
            raise ValueError(f"duplicate candidate skill_id: {skill_id}")
        if necessity not in VALID_NECESSITY:
            raise ValueError(f"invalid necessity for {skill_id}: {necessity!r}")
        try:
            relevance = float(item.get("relevance"))
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric score for {skill_id}") from exc
        if not 0.0 <= relevance <= 1.0 or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"score outside [0,1] for {skill_id}")
        seen.add(skill_id)
        normalized.append(
            {
                "skill_id": skill_id,
                "necessity": necessity,
                "relevance": relevance,
                "confidence": confidence,
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    missing = expected_skill_ids - seen
    if missing:
        raise ValueError(f"Judge omitted candidate skills: {sorted(missing)}")
    return sorted(normalized, key=lambda item: item["skill_id"])


def _cache_name(record_id: str) -> str:
    return _sha256_text(record_id)[:24] + ".json"


def _judge_one(
    task: dict[str, Any],
    candidates: list[dict[str, Any]],
    skills: dict[str, dict[str, Any]],
    cache_dir: Path,
    model: str,
    chat: ChatFunction,
    max_attempts: int,
) -> tuple[str, dict[str, Any], bool]:
    prompt = _build_prompt(task, candidates, skills)
    prompt_hash = _sha256_text(PROMPT_VERSION + "\n" + SYSTEM_PROMPT + "\n" + prompt)
    cache_path = cache_dir / _cache_name(task["record_id"])
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "success"
            and cached.get("model") == model
            and cached.get("prompt_version") == PROMPT_VERSION
            and cached.get("prompt_hash") == prompt_hash
        ):
            return task["record_id"], cached, True

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    attempts = []
    expected_ids = {row["skill_id"] for row in candidates}
    result: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = chat(payload)
            content = _extract_content(response)
            judgments = _parse_judgments(content, expected_ids)
            attempts.append({"attempt": attempt, "api_response": response, "parse_error": None})
            result = {
                "status": "success",
                "task_record_id": task["record_id"],
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash,
                "request": {
                    "model": model,
                    "messages": payload["messages"],
                    "temperature": 0,
                    "max_tokens": 4096,
                },
                "candidate_skill_ids": sorted(expected_ids),
                "judgments": judgments,
                "attempts": attempts,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            break
        except Exception as exc:
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 8))
    if result is None:
        result = {
            "status": "failed",
            "task_record_id": task["record_id"],
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "candidate_skill_ids": sorted(expected_ids),
            "attempts": attempts,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    _atomic_json(cache_path, result)
    return task["record_id"], result, False


def run_judge_pilot(
    tasks_path: Path | str,
    skills_path: Path | str,
    pilot_candidates_path: Path | str,
    selection_path: Path | str,
    output_root: Path | str,
    model: str,
    chat: ChatFunction,
    *,
    max_workers: int = 2,
    max_attempts: int = 3,
    limit: int | None = None,
    progress: ProgressFunction | None = None,
    collection_name: str = "pilot",
    fallback_cache_dirs: list[Path | str] | None = None,
) -> dict[str, Any]:
    """Judge a selected task collection and preserve raw evidence.

    ``collection_name`` isolates reports, evidence and response caches so a
    larger collection never overwrites the original pilot. Valid responses
    can optionally be imported from earlier cache directories.
    """
    if max_workers < 1 or max_attempts < 1:
        raise ValueError("max_workers and max_attempts must be positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", collection_name):
        raise ValueError("collection_name must contain only lowercase letters, digits, _ or -")
    tasks_path = Path(tasks_path).expanduser().resolve()
    skills_path = Path(skills_path).expanduser().resolve()
    pilot_candidates_path = Path(pilot_candidates_path).expanduser().resolve()
    selection_path = Path(selection_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    tasks_by_id = {row["record_id"]: row for row in _load_jsonl(tasks_path)}
    skills = {row["skill_id"]: row for row in _load_jsonl(skills_path)}
    candidates_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in _load_jsonl(pilot_candidates_path):
        candidates_by_task.setdefault(row["task_record_id"], []).append(row)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["record_ids"]
    selected_ids = selection[:limit] if limit is not None else selection
    missing = [record_id for record_id in selected_ids if record_id not in candidates_by_task]
    if missing:
        raise ValueError(f"selected tasks missing candidate rows: {missing[:5]}")
    unknown_skills = sorted(
        {
            row["skill_id"]
            for record_id in selected_ids
            for row in candidates_by_task[record_id]
            if row["skill_id"] not in skills
        }
    )
    if unknown_skills:
        raise ValueError(f"candidate skills missing from skills table: {unknown_skills}")

    cache_dir = output_root / "judge" / collection_name / "responses"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fallback_dirs = [Path(path).expanduser().resolve() for path in (fallback_cache_dirs or [])]
    # Import only cache entries whose model, prompt version and prompt hash
    # still validate inside _judge_one.
    for record_id in selected_ids:
        destination = cache_dir / _cache_name(record_id)
        if destination.exists():
            continue
        for fallback_dir in fallback_dirs:
            source = fallback_dir / _cache_name(record_id)
            if source.exists():
                try:
                    cached = json.loads(source.read_text(encoding="utf-8"))
                    if cached.get("status") == "success":
                        _atomic_json(destination, cached)
                        break
                except (OSError, json.JSONDecodeError):
                    continue
    results: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _judge_one,
                tasks_by_id[record_id],
                candidates_by_task[record_id],
                skills,
                cache_dir,
                model,
                chat,
                max_attempts,
            ): record_id
            for record_id in selected_ids
        }
        for future in concurrent.futures.as_completed(futures):
            record_id, result, cache_hit = future.result()
            results[record_id] = result
            cache_hits += int(cache_hit)
            completed += 1
            if progress:
                progress(completed, len(selected_ids), record_id, result["status"])

    evidence_rows = []
    failure_rows = []
    label_counts: Counter[str] = Counter()
    total_usage: Counter[str] = Counter()
    for record_id in selected_ids:
        result = results[record_id]
        if result["status"] != "success":
            failure_rows.append(result)
            continue
        successful_attempt = next(
            (attempt for attempt in reversed(result.get("attempts", [])) if "api_response" in attempt),
            {},
        )
        usage = (successful_attempt.get("api_response") or {}).get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                total_usage[key] += int(usage.get(key, 0) or 0)
            except (TypeError, ValueError):
                pass
        for judgment in result["judgments"]:
            label_counts[judgment["necessity"]] += 1
            evidence_rows.append(
                {
                    "evidence_id": "judge:" + _sha256_text(
                        f"{PROMPT_VERSION}:{model}:{record_id}:{judgment['skill_id']}"
                    )[:24],
                    "task_record_id": record_id,
                    "skill_id": judgment["skill_id"],
                    "label_source": "llm_judge",
                    "raw_label": judgment["necessity"],
                    "score": judgment["relevance"],
                    "confidence": judgment["confidence"],
                    "reason": judgment["reason"],
                    "source_model": model,
                    "source_prompt_version": PROMPT_VERSION,
                    "expert_plan_visible": True,
                    "candidate_only_before_judging": True,
                    "created_at": result["created_at"],
                }
            )

    labels_dir = output_root / "labels"
    reports_dir = output_root / "reports"
    evidence_path = labels_dir / f"{collection_name}_judge_evidence.jsonl"
    failures_path = labels_dir / f"{collection_name}_judge_failures.jsonl"
    evidence_count = _atomic_jsonl(evidence_path, evidence_rows)
    failure_count = _atomic_jsonl(failures_path, failure_rows)
    report = {
        "schema_version": "skilldag_ncf.judge_collection_report.v1",
        "collection_name": collection_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "task_count": len(selected_ids),
        "successful_task_count": len(selected_ids) - failure_count,
        "failed_task_count": failure_count,
        "cache_hit_task_count": cache_hits,
        "new_api_task_count": len(selected_ids) - cache_hits,
        "evidence_count": evidence_count,
        "label_counts": dict(sorted(label_counts.items())),
        "usage": dict(total_usage),
        "is_gold": False,
        "final_training_labels_generated": 0,
        "outputs": {
            "response_cache": str(cache_dir),
            "evidence": str(evidence_path),
            "failures": str(failures_path),
        },
    }
    report_path = reports_dir / f"judge_{collection_name}_report.json"
    report["outputs"]["report"] = str(report_path)
    _atomic_json(report_path, report)
    return report
