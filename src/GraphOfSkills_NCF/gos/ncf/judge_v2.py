"""Resumable one-call groupwise Judge with local task-evidence validation."""
from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .data_v2 import DATASET_VERSION, atomic_json, atomic_jsonl, load_jsonl
from .prompt_v2 import GENERIC_GROUPWISE_JUDGE_PROMPT

PROMPT_VERSION = "generic_groupwise_task_evidence_v2"
GRADES = {"primary": 2, "support": 1, "irrelevant": 0, "harmful": 0}
TARGETS = {"primary": 1.0, "support": 0.7, "irrelevant": 0.0, "harmful": 0.0}
SAMPLE_WEIGHTS = {
    "primary": 1.0,
    "support": 0.7,
    "irrelevant": 0.7,
    "harmful": 1.0,
}
SYSTEM_PROMPT = GENERIC_GROUPWISE_JUDGE_PROMPT


def _content(response: dict[str, Any]) -> str:
    return str(response["choices"][0]["message"]["content"])


def _normalize_evidence(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def validate_task_evidence(task_text: str, evidence: str) -> bool:
    """Evidence must be a non-empty contiguous quote from the raw task."""
    normalized_task = _normalize_evidence(task_text)
    normalized_evidence = _normalize_evidence(evidence)
    return bool(normalized_evidence and normalized_evidence in normalized_task)


def _parse(text: str, expected: set[str], task_text: str) -> dict[str, Any]:
    if text.strip().startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        fragment = text[text.find("{"):text.rfind("}") + 1]
        try:
            value = json.loads(fragment)
        except json.JSONDecodeError:
            value = ast.literal_eval(fragment)

    assigned: dict[str, str] = {}
    task_evidence: dict[str, str] = {}
    evidence_valid: dict[str, bool] = {}
    primary_values = value.get("primary_skills", [])
    if not isinstance(primary_values, list):
        raise ValueError("primary_skills must be a list")
    for item in primary_values:
        if not isinstance(item, dict):
            raise ValueError("each primary_skills item must be an object")
        skill_id = str(item.get("skill_id", "")).strip()
        quote = str(item.get("task_evidence", "")).strip()
        if skill_id not in expected or skill_id in assigned:
            raise ValueError(f"invalid or duplicate primary skill: {skill_id}")
        valid = validate_task_evidence(task_text, quote)
        assigned[skill_id] = "primary" if valid else "uncertain"
        task_evidence[skill_id] = quote
        evidence_valid[skill_id] = valid

    for field, role in {
        "support_skill_ids": "support",
        "harmful_skill_ids": "harmful",
        "uncertain_skill_ids": "uncertain",
    }.items():
        values = value.get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        for raw_skill_id in values:
            skill_id = str(raw_skill_id).strip()
            if skill_id not in expected or skill_id in assigned:
                raise ValueError(f"invalid or multiply assigned skill: {skill_id}")
            assigned[skill_id] = role

    primary = sorted(skill_id for skill_id, role in assigned.items() if role == "primary")
    if len(primary) > 3:
        raise ValueError("more than 3 evidence-valid primary skills")
    signature = value.get("task_signature", {})
    if not isinstance(signature, dict):
        raise ValueError("task_signature must be an object")
    rows = []
    for skill_id in sorted(expected):
        role = assigned.get(skill_id, "irrelevant")
        rows.append({
            "skill_id": skill_id,
            "role": role,
            "task_evidence": task_evidence.get(skill_id, ""),
            "task_evidence_valid": evidence_valid.get(skill_id),
            "quality_flags": (
                ["invalid_primary_task_evidence"]
                if skill_id in evidence_valid and not evidence_valid[skill_id] else []
            ),
        })
    return {
        "judgments": rows,
        "primary_skill_ids": primary,
        "task_signature": signature,
        "minimal_set_reason": str(value.get("minimal_set_reason", "")),
    }


def _build_payload(
    task: dict[str, Any],
    candidates: list[dict[str, Any]],
    skills: dict[str, dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    blocks = [{
        "skill_id": row["skill_id"],
        "description": skills[row["skill_id"]].get("description", ""),
        "body_preview": re.sub(
            r"\s+", " ", skills[row["skill_id"]].get("skill_body", "")
        )[:600],
        "retrieval_sources": row.get("candidate_sources", []),
    } for row in candidates]
    prompt = (
        "RAW_TASK:\n"
        + json.dumps(
            {"task_id": task["record_id"], "task_text": task["task_text"]},
            ensure_ascii=False,
        )
        + "\nCANDIDATES:\n"
        + json.dumps(blocks, ensure_ascii=False)
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }


def _judge_one(
    task: dict[str, Any],
    candidates: list[dict[str, Any]],
    skills: dict[str, dict[str, Any]],
    model: str,
    chat: Callable[[dict[str, Any]], dict[str, Any]],
    cache_dir: Path,
    max_attempts: int,
    force: bool,
) -> tuple[str, dict[str, Any], bool]:
    payload = _build_payload(task, candidates, skills, model)
    prompt_hash = hashlib.sha256(
        (PROMPT_VERSION + json.dumps(payload["messages"], ensure_ascii=False)).encode()
    ).hexdigest()
    cache_path = cache_dir / (hashlib.sha256(task["record_id"].encode()).hexdigest()[:24] + ".json")
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("status") == "success"
            and cached.get("model") == model
            and cached.get("prompt_version") == PROMPT_VERSION
            and cached.get("prompt_hash") == prompt_hash
        ):
            return task["record_id"], cached, True
    expected = {row["skill_id"] for row in candidates}
    attempts = []
    result = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = chat(payload)
            parsed = _parse(_content(response), expected, task["task_text"])
            result = {
                "status": "success",
                "task_record_id": task["record_id"],
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash,
                "parsed": parsed,
                "usage": response.get("usage", {}),
            }
            break
        except Exception as exc:
            attempts.append(f"{type(exc).__name__}: {exc}")
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 8))
    if result is None:
        result = {
            "status": "failed",
            "task_record_id": task["record_id"],
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "errors": attempts,
        }
    atomic_json(cache_path, result)
    return task["record_id"], result, False


def run_gos_v2_judge(
    tasks_path: Path | str,
    skills_path: Path | str,
    candidates_path: Path | str,
    selection_path: Path | str,
    output_root: Path | str,
    model: str,
    chat: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    collection_name: str = "generic_pilot",
    max_workers: int = 4,
    max_attempts: int = 3,
    limit: int | None = None,
    force_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", collection_name):
        raise ValueError("invalid collection_name")
    tasks = {row["record_id"]: row for row in load_jsonl(Path(tasks_path))}
    skills = {row["skill_id"]: row for row in load_jsonl(Path(skills_path))}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(Path(candidates_path)):
        grouped[row["task_record_id"]].append(row)
    selected_ids = json.loads(Path(selection_path).read_text())["record_ids"]
    if limit is not None:
        selected_ids = selected_ids[:limit]
    missing = [task_id for task_id in selected_ids if task_id not in tasks or task_id not in grouped]
    if missing:
        raise ValueError(f"selection contains missing tasks/candidates: {missing[:3]}")

    out = Path(output_root) / "data"
    cache_dir = out / "judge" / collection_name / "responses"
    cache_dir.mkdir(parents=True, exist_ok=True)
    force_task_ids = force_task_ids or set()
    results: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _judge_one, tasks[task_id], grouped[task_id], skills, model, chat,
                cache_dir, max_attempts, task_id in force_task_ids,
            ): task_id
            for task_id in selected_ids
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task_id, result, cache_hit = future.result()
            results[task_id] = result
            cache_hits += int(cache_hit)
            print(f"[v2-judge] {done}/{len(futures)} {result['status']} {task_id}", flush=True)

    evidence, failures = [], []
    roles: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    invalid_primary_evidence = 0
    for task_id in selected_ids:
        result = results[task_id]
        if result["status"] != "success":
            failures.append(result)
            continue
        parsed = result["parsed"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int(result.get("usage", {}).get(key, 0) or 0)
        for row in parsed["judgments"]:
            role = row["role"]
            roles[role] += 1
            invalid_primary_evidence += int("invalid_primary_task_evidence" in row["quality_flags"])
            evidence.append({
                "dataset_version": DATASET_VERSION,
                "task_record_id": task_id,
                "skill_id": row["skill_id"],
                "role": role,
                "target_grade": GRADES.get(role),
                "target_score": TARGETS.get(role),
                "trainable": role in GRADES,
                # Sample weight is confidence/importance, not the relevance
                # target.  A zero weight here would silently remove negative
                # examples from the training loss.
                "sample_weight": SAMPLE_WEIGHTS.get(role),
                "task_evidence": row["task_evidence"],
                "task_evidence_valid": row["task_evidence_valid"],
                "quality_flags": row["quality_flags"],
                "primary_skill_ids": parsed["primary_skill_ids"],
                "task_signature": parsed["task_signature"],
                "minimal_set_reason": parsed["minimal_set_reason"],
                "source_model": model,
                "source_prompt_version": PROMPT_VERSION,
                "expert_plan_visible": False,
            })

    labels_path = out / "labels" / f"task_skill_v2_gos_{collection_name}.jsonl"
    failures_path = out / "judge" / collection_name / "failures.jsonl"
    report_path = out / "reports" / f"task_skill_v2_gos_{collection_name}_report.json"
    atomic_jsonl(labels_path, evidence)
    atomic_jsonl(failures_path, failures)
    primary_frequency = Counter(row["skill_id"] for row in evidence if row["role"] == "primary")
    successful = len({row["task_record_id"] for row in evidence})
    report = {
        "dataset_version": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "selected_task_count": len(selected_ids),
        "successful_task_count": successful,
        "failed_task_count": len(failures),
        "cache_hit_task_count": cache_hits,
        "role_counts": dict(roles),
        "invalid_primary_evidence_count": invalid_primary_evidence,
        "usage": dict(usage),
        "primary_skill_frequency": dict(primary_frequency.most_common()),
        "max_primary_task_rate": (
            max(primary_frequency.values()) / successful if primary_frequency else 0.0
        ),
        "expert_plan_visible": False,
        "outputs": {
            "labels": str(labels_path),
            "failures": str(failures_path),
            "cache_dir": str(cache_dir),
        },
    }
    atomic_json(report_path, report)
    return report
