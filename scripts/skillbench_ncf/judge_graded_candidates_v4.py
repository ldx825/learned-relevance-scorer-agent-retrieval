#!/usr/bin/env python3
"""Grade a fixed SkillsBench candidate pilot with one request per query."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_evidence_v1 import ROOT, dump_json, dump_jsonl, load_jsonl
from naturalize_minimax_pilot_v1 import (
    MODEL,
    extract_content,
    parse_json_object,
    post_json,
)


MANIFEST = ROOT / "data/skillbench_ncf/graded_judge_pilot_v4/judge_manifest.jsonl"
OUTPUT_DIR = ROOT / "data/skillbench_ncf/graded_judge_pilot_v4/judgments"
REPORT = (
    ROOT
    / "artifacts/skillbench_ncf/manifests/graded_judge_pilot_v4_results.json"
)
REPAIRS = (
    ROOT
    / "configs/skillbench_ncf/graded_judge_pilot_v4_repairs.json"
)
PROMPT_VERSION = "skillbench_ncf_graded_candidate_judge_v4"

SYSTEM_PROMPT = """You are a conservative offline annotator for skill retrieval.
Judge each candidate only for the supplied user query and only from the supplied
skill evidence. Do not infer undocumented capabilities from a service name.

Grades:
- 2: directly performs a core operation required by the query. Be conservative.
  Multiple grade-2 skills are allowed only when they provide distinct core
  capabilities, not interchangeable or merely similar alternatives.
- 1: useful complementary support, prerequisite, verification, or recovery,
  but it does not directly perform the core requested operation.
- 0: irrelevant, wrong service/operation, or offers no concrete help.
- uncertain: the supplied evidence is insufficient or genuinely ambiguous.

A graph relation or retrieval rank is not evidence of usefulness. Candidate
sources are supplied only for later audit and must not determine the grade.
Do not output numerical confidence. Return exactly one JSON object:
{
  "query_id": "exact input query_id",
  "judgments": [
    {
      "skill_id": "exact candidate skill_id",
      "grade": 2,
      "evidence": "short supplied capability phrase",
      "reason_zh": "简短中文理由"
    }
  ]
}
Return every candidate exactly once. grade must be 0, 1, 2, or "uncertain"."""


def api_payload(row: dict[str, Any]) -> str:
    # Deliberately omit audit_source_skill_id, source-anchor status and any
    # known-source flag. The Judge must use capability evidence.
    payload = {
        "query_id": row["query_id"],
        "query": row["query_text"],
        "candidates": [
            {
                "skill_id": candidate["skill_id"],
                "description": candidate["description"],
                "local_operation_evidence": candidate[
                    "local_operation_evidence"
                ],
                "linked_public_operation_evidence": candidate[
                    "linked_public_operation_evidence"
                ],
            }
            for candidate in row["candidates"]
        ],
    }
    return "Grade this candidate set:\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    )


def normalize(
    payload: dict[str, Any],
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    raw = payload.get("judgments")
    if not isinstance(raw, list):
        raise ValueError("response has no judgments list")
    expected = {candidate["skill_id"] for candidate in row["candidates"]}
    repair_config = json.loads(REPAIRS.read_text(encoding="utf-8"))
    repair = repair_config.get("queries", {}).get(row["query_id"], {})
    aliases = repair.get("skill_id_aliases", {})
    returned_query_id = str(payload.get("query_id", ""))
    query_id_mismatch = returned_query_id != row["query_id"]
    returned_expected_ids = {
        str(aliases.get(str(item.get("skill_id", "")).strip(), str(item.get("skill_id", "")).strip()))
        for item in raw
        if isinstance(item, dict)
    } & expected
    if query_id_mismatch and len(returned_expected_ids) / len(expected) < 0.8:
        raise ValueError(
            "response query_id mismatch with insufficient candidate overlap"
        )
    seen: set[str] = set()
    result = []
    for item in raw:
        returned_skill_id = str(item.get("skill_id", "")).strip()
        skill_id = str(aliases.get(returned_skill_id, returned_skill_id))
        # Never assign a hallucinated or duplicate ID to an expected
        # candidate. Ignore it here; the genuinely unreturned expected item is
        # conservatively materialized as uncertain below.
        if skill_id not in expected or skill_id in seen:
            continue
        seen.add(skill_id)
        grade: int | str = item.get("grade")
        if isinstance(grade, str):
            grade = grade.strip().lower()
            if grade in {"0", "1", "2"}:
                grade = int(grade)
        if grade not in {0, 1, 2, "uncertain"}:
            raise ValueError(f"invalid grade for {skill_id}: {grade!r}")
        result.append(
            {
                "schema_version": "skillbench_ncf.graded_candidate_label.v4",
                "pair_id": f"{row['query_id']}::{skill_id}",
                "query_id": row["query_id"],
                "candidate_skill_id": skill_id,
                "raw_judge_grade": grade,
                "judge_evidence": str(item.get("evidence", "")).strip()[:800],
                "judge_reason_zh": str(item.get("reason_zh", "")).strip()[:800],
                "judge_model": MODEL,
                "judge_prompt_version": PROMPT_VERSION,
                "official_skillsbench_task_used": False,
                "training_ready": False,
                "normalization_action": (
                    "unambiguous_skill_id_alias"
                    if returned_skill_id != skill_id
                    else (
                        "query_id_mismatch_with_candidate_overlap"
                        if query_id_mismatch
                        else "none"
                    )
                ),
                "returned_skill_id": returned_skill_id,
                "returned_query_id": returned_query_id,
            }
        )
    missing = expected - seen
    for skill_id in sorted(missing):
        result.append(
            {
                "schema_version": "skillbench_ncf.graded_candidate_label.v4",
                "pair_id": f"{row['query_id']}::{skill_id}",
                "query_id": row["query_id"],
                "candidate_skill_id": skill_id,
                "raw_judge_grade": "uncertain",
                "judge_evidence": "",
                "judge_reason_zh": (
                    "模型未返回该候选的有效唯一ID；按保守规则标为uncertain，"
                    "不推断数值标签。"
                ),
                "judge_model": MODEL,
                "judge_prompt_version": PROMPT_VERSION,
                "official_skillsbench_task_used": False,
                "training_ready": False,
                "normalization_action": "missing_response_to_uncertain",
                "returned_skill_id": None,
                "returned_query_id": returned_query_id,
            }
        )
    return sorted(result, key=lambda item: item["candidate_skill_id"])


def call_one(
    row: dict[str, Any],
    output_dir: Path,
    api_base: str,
    api_key: str,
    timeout: int,
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    user_prompt = api_payload(row)
    prompt_hash = "sha256:" + hashlib.sha256(
        (
            PROMPT_VERSION
            + "\n"
            + SYSTEM_PROMPT
            + "\n"
            + user_prompt
        ).encode("utf-8")
    ).hexdigest()
    path = output_dir / "responses" / f"{stable_name(row['query_id'])}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("status") in {"success", "response_received"}
            and cached.get("model") == MODEL
            and cached.get("prompt_hash") == prompt_hash
        ):
            response = cached["response"]
            usage = response.get("usage") or {}
            normalized = normalize(
                parse_json_object(extract_content(response)), row
            )
            if cached.get("status") != "success":
                cached["status"] = "success"
                cached["normalized_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                dump_json(path, cached)
            return (
                normalized,
                True,
                {
                    key: int(usage.get(key, 0) or 0)
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    )
                },
            )
        raise RuntimeError(f"stale response cache requires inspection: {path}")

    response = post_json(
        api_base.rstrip("/") + "/chat/completions",
        api_key,
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 8000,
        },
        timeout,
    )
    # Cache the paid response before parsing. A schema error must not cause a
    # second paid request on resume.
    dump_json(
        path,
        {
            "status": "response_received",
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": response,
        },
    )
    normalized = normalize(parse_json_object(extract_content(response)), row)
    cached_payload = json.loads(path.read_text(encoding="utf-8"))
    cached_payload["status"] = "success"
    cached_payload["normalized_at"] = datetime.now(timezone.utc).isoformat()
    dump_json(path, cached_payload)
    usage = response.get("usage") or {}
    return (
        normalized,
        False,
        {
            key: int(usage.get(key, 0) or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    )


def stable_name(query_id: str) -> str:
    return hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:20]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent API requests. Does not change prompts or labels.",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("SKILLDAG_LLM_BASE", "https://yunwu.ai/v1"),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("SKILLDAG_LLM_TIMEOUT_S", "240")),
    )
    args = parser.parse_args()
    args.manifest = (
        args.manifest
        if args.manifest.is_absolute()
        else (ROOT / args.manifest).resolve()
    )
    args.output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else (ROOT / args.output_dir).resolve()
    )
    args.report = (
        args.report
        if args.report.is_absolute()
        else (ROOT / args.report).resolve()
    )
    api_key = os.environ.get("SKILLDAG_LLM_API_KEY", "")
    if not api_key:
        raise SystemExit("SKILLDAG_LLM_API_KEY is empty")

    rows = load_jsonl(args.manifest)
    if not 1 <= args.limit <= len(rows):
        raise ValueError("--limit is outside manifest bounds")
    if not 1 <= args.workers <= 160:
        raise ValueError("--workers must be between 1 and 160")
    rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "responses").mkdir(parents=True, exist_ok=True)

    labels: list[dict[str, Any]] = []
    usage = Counter()
    cache_hits = 0
    api_request_attempts = 0
    failures = []
    work = []
    for index, row in enumerate(rows, start=1):
        response_path = (
            args.output_dir
            / "responses"
            / f"{stable_name(row['query_id'])}.json"
        )
        request_needed = not response_path.exists()
        api_request_attempts += int(request_needed)
        work.append((index, row))

    def execute(item: tuple[int, dict[str, Any]]) -> tuple[
        int,
        dict[str, Any],
        list[dict[str, Any]],
        bool,
        dict[str, int],
    ]:
        index, row = item
        result, cached, item_usage = call_one(
            row,
            args.output_dir,
            args.api_base,
            api_key,
            args.timeout,
        )
        return index, row, result, cached, item_usage

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_item = {
            executor.submit(execute, item): item for item in work
        }
        for future in as_completed(future_to_item):
            index, row = future_to_item[future]
            completed += 1
            try:
                _, _, result, cached, item_usage = future.result()
                labels.extend(result)
                cache_hits += int(cached)
                usage.update(item_usage)
                print(
                    f"[{completed}/{len(rows)}] source_index={index} "
                    f"{row['query_id']} labels={len(result)} "
                    f"cached={int(cached)}",
                    flush=True,
                )
            except Exception as exc:  # preserve paid responses for resumption
                failures.append(
                    {
                        "query_id": row["query_id"],
                        "source_index": index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"[{completed}/{len(rows)}] FAILED source_index={index} "
                    f"{row['query_id']}: {exc}",
                    flush=True,
                )

    labels.sort(key=lambda row: (row["query_id"], row["candidate_skill_id"]))
    failures.sort(key=lambda row: row["source_index"])
    dump_jsonl(args.output_dir / "labels.jsonl", labels)
    dump_jsonl(args.output_dir / "failures.jsonl", failures)
    grade_counts = Counter(str(row["raw_judge_grade"]) for row in labels)
    normalization_counts = Counter(
        row["normalization_action"] for row in labels
    )
    labels_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in labels:
        labels_by_query.setdefault(row["query_id"], []).append(row)
    source_grade_counts = Counter()
    no_grade2 = []
    for manifest_row in rows:
        query_labels = labels_by_query.get(manifest_row["query_id"])
        if not query_labels:
            continue
        by_skill = {
            row["candidate_skill_id"]: row["raw_judge_grade"]
            for row in query_labels
        }
        source_grade_counts[str(by_skill[manifest_row["audit_source_skill_id"]])] += 1
        if 2 not in by_skill.values():
            no_grade2.append(manifest_row["query_id"])

    report = {
        "schema_version": "skillbench_ncf.graded_judge_pilot_results.v4",
        "status": "complete" if not failures else "partial",
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "official_skillsbench_tasks_used": False,
        "api_request_attempts_this_run": api_request_attempts,
        "new_cached_api_responses_this_run": sum(
            1
            for row in rows
            if (
                args.output_dir
                / "responses"
                / f"{stable_name(row['query_id'])}.json"
            ).exists()
        )
        - cache_hits,
        "cached_api_responses_cumulative": sum(
            1
            for row in rows
            if (
                args.output_dir
                / "responses"
                / f"{stable_name(row['query_id'])}.json"
            ).exists()
        ),
        "cache_hits": cache_hits,
        "counts": {
            "queries_requested": len(rows),
            "queries_completed": len(labels_by_query),
            "candidate_labels": len(labels),
            "failures": len(failures),
            "grade_counts": dict(sorted(grade_counts.items())),
            "normalization_actions": dict(
                sorted(normalization_counts.items())
            ),
            "audit_source_grade_counts": dict(sorted(source_grade_counts.items())),
            "queries_without_grade2": len(no_grade2),
        },
        "usage_cumulative_from_cached_responses": dict(usage),
        "queries_without_grade2": no_grade2,
        "limitations": [
            "MiniMax-M2.7 is also the downstream agent model; same-model systematic bias may remain.",
            "This pilot is a quality-control sample, not the final training corpus.",
            "Numeric confidence is deliberately excluded after prior overconfidence.",
            "Labels remain training_ready=false until deterministic calibration and audit pass.",
            "Explicit deterministic repairs are declared in configs/skillbench_ncf/graded_judge_pilot_v4_repairs.json.",
            "Any unreturned, hallucinated-ID, or duplicate-ID candidate is conservatively normalized to uncertain rather than assigned a numeric grade.",
            "A mismatched returned query_id is accepted only when at least 80% of returned candidate IDs belong to the expected candidate set.",
        ],
        "private_outputs": {
            "labels": str((args.output_dir / "labels.jsonl").relative_to(ROOT)),
            "failures": str((args.output_dir / "failures.jsonl").relative_to(ROOT)),
            "responses": str((args.output_dir / "responses").relative_to(ROOT)),
        },
    }
    dump_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
