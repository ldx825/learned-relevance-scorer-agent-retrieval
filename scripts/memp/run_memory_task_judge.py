#!/usr/bin/env python3
"""Resumable MiniMax Judge for train-only Memory-Task candidate batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src/SkillDAG_NCF"
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from skilldag.initialize import _http_post_json  # noqa: E402


PROMPT_VERSION = "task_resource_ncf_memory_judge_v4"
QUERY_ONLY_PROMPT_VERSION = "task_resource_ncf_memory_query_judge_v5"
VALID_GRADES: set[int | str] = {0, 1, 2, "uncertain"}
RELATION_VALUES = {"match", "partial", "conflict", "not_applicable"}
VALIDITY_VALUES = {"valid", "invalid", "uncertain"}
SYSTEM_PROMPT = """You are an offline annotator for procedural-memory retrieval in an embodied ALFWorld agent.

Judge how useful each candidate MEMORY is for completing the given TASK. Judge the actual workflow, not merely shared words in the memory query.

Unified Task-Resource grades:
- 2 (Primary/required): this one memory provides a correct, executable procedure that directly covers all goal-critical task requirements.
- 1 (Support/helpful): it provides a distinct and correct complementary subprocedure, prerequisite, verification, or recovery step, but is insufficient by itself.
- 0 (Irrelevant/harmful): it is irrelevant, conflicts with the required operation/cardinality/destination, is procedurally invalid, is merely redundant, or may mislead the agent.
- uncertain: the supplied evidence is genuinely insufficient or ambiguous.

Critical distinction for grade 1:
- Grade 1 may be either a task-specific complementary subprocedure OR a transferable, operation-specific procedure whose steps remain valid after substituting the concrete object/destination.
- A broad topical or lexical match is not enough. The workflow must provide actionable steps that solve a real part of THIS task.
- If both object and destination conflict, grade 1 is allowed only when the memory supplies a valid, reusable operation-level procedure that is genuinely useful for THIS task (for example sequential two-object handling or a correct heat/cool/clean/tool workflow). Otherwise grade 0.
- If multiple memories supply the same interchangeable support role, select only the single most task-specific representative as grade 1 and grade the redundant alternatives 0.
- A memory with the same operation and object but a different destination may support the operation subprocedure. A memory with the same operation and destination but a different object may support the destination/tool subprocedure. Within each such support role, keep at most one representative.
- A generic operation-level template is a separate support role. Keep at most one such representative, and prefer a valid workflow with matching cardinality and genuinely transferable steps over lexical similarity.

ALFWorld constraints:
- clean, heat, and cool are different operations; an operation conflict is grade 0.
- a two-object task requires completing pick-and-place twice. A single-object memory can be grade 1 if it gives a valid subprocedure, never grade 2.
- the agent normally carries one movable object at a time. A workflow requiring it to collect/hold both objects before placing is invalid.
- matching only an object or destination does not justify grade 2.
- do not assume actions, tools, locations, or ordering absent from the workflow.

For every candidate, explicitly assess operation, object, cardinality, destination, procedural validity, and whether it adds a distinct contribution. Confidence is evidence strength, not a habitual constant; use 0.90+ only for explicit unambiguous evidence.

Use ONLY these exact enum values:
- operation/object/cardinality/destination: "match", "partial", "conflict", or "not_applicable"
- procedural_validity: "valid", "invalid", or "uncertain"

Return exactly one JSON object with no markdown and every memory exactly once:
{"judgments":[{"memory_id":0,"grade":2,"confidence":0.91,"operation":"match","object":"match","cardinality":"match","destination":"match","procedural_validity":"valid","distinct_contribution":"complete procedure","reason":"short evidence-based reason"}]}

grade must be 0, 1, 2, or "uncertain". confidence must be in [0,1]. Before returning, compare all memories as a set and remove redundant grade-1 labels."""

QUERY_ONLY_SYSTEM_PROMPT = """You are an offline annotator for query-based procedural-memory retrieval in an embodied ALFWorld agent.

You are given a train-only TASK QUERY (sometimes with a retrieval focus) and candidate MEMORY SOURCE QUERIES. The workflow body is intentionally hidden because both the paper Query-cosine baseline and the Content-NeuMF plug-in rank only query embeddings.

Unified Task-Resource grades:
- 2 (Primary/required): the memory source query directly matches the current retrieval focus, including every constraint stated by that focus.
- 1 (Support/helpful): it supplies a distinct complementary relation needed by the full task, such as the correct operation/cardinality, object-handling pattern, or destination/tool pattern, but is insufficient for the current focus by itself.
- 0 (Irrelevant/harmful): it conflicts with the required operation or cardinality, only shares incidental words, duplicates another stronger candidate's role, or does not help the full task.
- uncertain: the two queries genuinely do not provide enough evidence.

Rules:
- clean, heat, cool, examine, one-object placement, and two-object placement are different operations.
- two-object handling must not be treated as equivalent to one-object handling.
- operation agreement is more important than object/destination word overlap for an operation-focused query.
- object identity and cardinality dominate an object-focused query.
- destination/tool identity and cardinality dominate a destination-focused query.
- for the unmodified full task, grade 2 requires operation, object, cardinality and destination/tool to match.
- keep at most one interchangeable grade-1 representative per support role.
- because no workflow is supplied, procedural_validity must be "uncertain"; do not invent workflow steps.

For every candidate return exactly these fields and enum values:
- operation/object/cardinality/destination: "match", "partial", "conflict", or "not_applicable"
- procedural_validity: "uncertain"

Return exactly one JSON object with no markdown and every memory exactly once:
{"judgments":[{"memory_id":0,"grade":2,"confidence":0.91,"operation":"match","object":"match","cardinality":"match","destination":"match","procedural_validity":"uncertain","distinct_contribution":"direct query match","reason":"short query-evidence reason"}]}

grade must be 0, 1, 2, or "uncertain". confidence must be in [0,1]. Compare the candidates as a set before returning."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prompt_for(
    batch: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
    *,
    evidence_mode: str = "query_plus_workflow",
) -> str:
    visible_candidates = candidates if candidates is not None else batch["candidates"]
    # Deliberately construct a fresh whitelist instead of copying candidate
    # dictionaries, so rule pseudo-labels cannot leak into Judge prompts.
    if evidence_mode == "query_only":
        memory_blocks = [
            {"memory_id": int(row["memory_id"]), "memory_query": row["memory_query"]}
            for row in visible_candidates
        ]
    else:
        memory_blocks = [
            {
                "memory_id": int(row["memory_id"]),
                "memory_query": row["memory_query"],
                "workflow": row["workflow"],
            }
            for row in visible_candidates
        ]
    context = {
        "task_query": batch["task_query"],
        "task_signature": batch["task_signature"],
        "important": (
            "Judge only the task-query to memory-query relation; workflow evidence is unavailable."
            if evidence_mode == "query_only"
            else "Judge the candidate as procedural memory for this task; do not infer unstated workflow steps."
        ),
    }
    return "TASK_CONTEXT:\n" + json.dumps(context, ensure_ascii=False, sort_keys=True) + "\n\nMEMORIES:\n" + json.dumps(memory_blocks, ensure_ascii=False, sort_keys=True)


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def parse_response(content: str, expected_ids: set[int]) -> list[dict[str, Any]]:
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
    required_text_fields = (
        "operation",
        "object",
        "cardinality",
        "destination",
        "procedural_validity",
        "distinct_contribution",
        "reason",
    )
    for item in judgments:
        memory_id = int(item.get("memory_id"))
        grade: int | str = item.get("grade")
        if isinstance(grade, str):
            grade = grade.strip().lower()
            if grade in {"0", "1", "2"}:
                grade = int(grade)
        if memory_id not in expected_ids or memory_id in seen:
            raise ValueError(f"unknown or duplicate memory_id: {memory_id}")
        if grade not in VALID_GRADES:
            raise ValueError(f"invalid grade for memory_id={memory_id}: {grade!r}")
        confidence = float(item.get("confidence"))
        if not 0 <= confidence <= 1:
            raise ValueError(f"invalid confidence for memory_id={memory_id}")
        fields = {key: str(item.get(key, "")).strip() for key in required_text_fields}
        if any(not value for value in fields.values()):
            raise ValueError(f"missing evidence field for memory_id={memory_id}")
        for field in ("operation", "object", "cardinality", "destination"):
            if fields[field] not in RELATION_VALUES:
                raise ValueError(
                    f"invalid {field} enum for memory_id={memory_id}: {fields[field]!r}"
                )
        if fields["procedural_validity"] not in VALIDITY_VALUES:
            raise ValueError(
                "invalid procedural_validity enum for "
                f"memory_id={memory_id}: {fields['procedural_validity']!r}"
            )
        seen.add(memory_id)
        normalized.append({"memory_id": memory_id, "grade": grade, "confidence": confidence, **fields})
    if seen != expected_ids:
        raise ValueError(f"missing memory IDs: {sorted(expected_ids - seen)}")
    return sorted(normalized, key=lambda row: row["memory_id"])


def cache_name(task_id: int) -> str:
    return f"task_{task_id:04d}.json"


def request_once(prompt: str, expected_ids: set[int], args: argparse.Namespace) -> tuple[list[dict], dict]:
    request = {
        "model": args.model,
        "messages": [{"role": "system", "content": args.system_prompt}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    status, body = _http_post_json(
        args.api_base.rstrip("/") + "/chat/completions",
        {"Authorization": f"Bearer {args.api_key}"},
        request,
        args.timeout,
    )
    if status >= 400:
        raise RuntimeError(f"HTTP {status}: {body[:400]}")
    response = json.loads(body)
    return parse_response(extract_content(response), expected_ids), response


def judge_one(batch: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    prompt = prompt_for(batch, evidence_mode=args.evidence_mode)
    prompt_hash = "sha256:" + hashlib.sha256((args.prompt_version + "\n" + args.system_prompt + "\n" + prompt).encode()).hexdigest()
    cache_path = args.output_dir / "responses" / cache_name(int(batch["task_id"]))
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("status") == "success" and cached.get("model") == args.model and cached.get("prompt_hash") == prompt_hash:
            return cached, True

    attempts = []
    result = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            judgments, response = request_once(
                prompt,
                {int(row["memory_id"]) for row in batch["candidates"]},
                args,
            )
            attempts.append({"attempt": attempt, "response": response, "error": None})
            result = {
                "status": "success",
                "task_id": int(batch["task_id"]),
                "model": args.model,
                "prompt_version": args.prompt_version,
                "prompt_hash": prompt_hash,
                "judgments": judgments,
                "attempts": attempts,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            break
        except Exception as exc:
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})

    if result is None:
        result = {
            "status": "failed",
            "task_id": int(batch["task_id"]),
            "model": args.model,
            "prompt_version": args.prompt_version,
            "prompt_hash": prompt_hash,
            "attempts": attempts,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--model", default=os.environ.get("SKILLDAG_LLM_MODEL", "MiniMax-M2.7"))
    parser.add_argument("--api-base", default=os.environ.get("SKILLDAG_LLM_BASE", "https://yunwu.ai/v1"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--evidence-mode",
        choices=("query_only", "query_plus_workflow"),
        default="query_plus_workflow",
    )
    args = parser.parse_args()
    if args.evidence_mode == "query_only":
        args.prompt_version = QUERY_ONLY_PROMPT_VERSION
        args.system_prompt = QUERY_ONLY_SYSTEM_PROMPT
    else:
        args.prompt_version = PROMPT_VERSION
        args.system_prompt = SYSTEM_PROMPT

    batches = load_jsonl(args.candidates)
    selected = batches[args.offset : args.offset + args.limit if args.limit is not None else None]
    if args.dry_run:
        lengths = [
            len(args.system_prompt) + len(prompt_for(batch, evidence_mode=args.evidence_mode))
            for batch in selected
        ]
        report = {
            "status": "dry_run",
            "prompt_version": args.prompt_version,
            "task_count": len(selected),
            "candidate_pair_count": sum(len(row["candidates"]) for row in selected),
            "prompt_chars": {"min": min(lengths), "mean": sum(lengths) / len(lengths), "max": max(lengths)},
            "rule_seed_fields_visible_to_judge": False,
            "uses_eval_data": False,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    args.api_key = os.environ.get("SKILLDAG_LLM_API_KEY", "")
    if not args.api_key:
        raise SystemExit("SKILLDAG_LLM_API_KEY is empty")
    results = {}
    cache_hits = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(judge_one, batch, args): int(batch["task_id"]) for batch in selected}
        for index, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            result, cache_hit = future.result()
            results[int(result["task_id"])] = result
            cache_hits += int(cache_hit)
            print(f"[memory-judge] {index}/{len(selected)} {result['status']} cache={int(cache_hit)} task={result['task_id']}", flush=True)

    labels = []
    failures = []
    grades: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    for batch in selected:
        result = results[int(batch["task_id"])]
        if result["status"] != "success":
            failures.append(result)
            continue
        for attempt in result.get("attempts", []):
            response = attempt.get("response") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += int((response.get("usage") or {}).get(key, 0) or 0)
        for judgment in result["judgments"]:
            grades[str(judgment["grade"])] += 1
            labels.append({
                "task_id": int(batch["task_id"]),
                **judgment,
                "model": args.model,
                "prompt_version": args.prompt_version,
                "uses_eval_data": False,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "labels.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
    (args.output_dir / "failures.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in failures), encoding="utf-8")
    report = {
        "schema_version": "task_resource_ncf.memory_judge_report.v1",
        "model": args.model,
        "prompt_version": args.prompt_version,
        "task_count": len(selected),
        "successful_task_count": len(selected) - len(failures),
        "failed_task_count": len(failures),
        "judgment_count": len(labels),
        "grade_counts": dict(sorted(grades.items())),
        "cache_hit_count": cache_hits,
        "usage": dict(usage),
        "uses_eval_data": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
