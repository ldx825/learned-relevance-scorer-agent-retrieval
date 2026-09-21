#!/usr/bin/env python3
"""Build the full 3,650-query graded-Judge manifest without API calls."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from build_evidence_v1 import ROOT, dump_json, dump_jsonl, load_jsonl
from build_graded_judge_pilot_v4 import (
    CANDIDATES,
    ENRICHED,
    EVIDENCE,
    choose_candidates,
    evidence_summary,
)
from source_policy import assert_training_source, load_policy


OUTPUT_DIR = ROOT / "data/skillbench_ncf/graded_judge_full_v4"
MANIFEST = OUTPUT_DIR / "judge_manifest.jsonl"
REPORT = (
    ROOT
    / "artifacts/skillbench_ncf/manifests/graded_judge_full_v4_report.json"
)


def main() -> int:
    policy = load_policy()
    for path in (CANDIDATES, EVIDENCE, ENRICHED):
        assert_training_source(path, policy)

    candidates = load_jsonl(CANDIDATES)
    cards = {row["skill_id"]: row for row in load_jsonl(EVIDENCE)}
    enriched = {row["skill_id"]: row for row in load_jsonl(ENRICHED)}
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_query[str(row["query_id"])].append(row)

    manifests = []
    for query_id in sorted(by_query):
        rows = by_query[query_id]
        first = rows[0]
        selected = choose_candidates(rows)
        manifests.append(
            {
                "schema_version": "skillbench_ncf.graded_judge_query.v4",
                "query_id": query_id,
                "query_text": first["query_text"],
                "query_source_type": first["query_source_type"],
                "internal_family_split": first["internal_family_split"],
                "audit_source_skill_id": first["source_skill_id"],
                "audit_source_anchor_needed": any(
                    "source_anchor" in row["candidate_sources"] for row in rows
                ),
                "official_skillsbench_task_used": False,
                "candidate_count": len(selected),
                "candidates": [
                    {
                        **evidence_summary(
                            str(row["candidate_skill_id"]),
                            str(first["query_text"]),
                            cards,
                            enriched,
                        ),
                        "candidate_sources": row["candidate_sources"],
                    }
                    for row in selected
                ],
                "labels": None,
                "api_submission_authorized": True,
            }
        )

    if len(manifests) != 3650:
        raise AssertionError(f"expected 3650 queries, got {len(manifests)}")
    if any(row["candidate_count"] != 20 for row in manifests):
        raise AssertionError("every full-Judge query must keep 20 candidates")
    if any(row["official_skillsbench_task_used"] for row in manifests):
        raise AssertionError("official SkillsBench task leakage")
    if any(row["labels"] is not None for row in manifests):
        raise AssertionError("full manifest contains labels before Judge")

    dump_jsonl(MANIFEST, manifests)
    prompt_chars = 0
    # Match judge_graded_candidates_v4.api_payload without importing its API
    # machinery, so this remains an explicitly zero-API construction step.
    for row in manifests:
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
        prompt_chars += len(
            "Grade this candidate set:\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    report = {
        "schema_version": "skillbench_ncf.graded_judge_full_report.v4",
        "status": "manifest_complete_zero_api",
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "counts": {
            "queries": len(manifests),
            "candidate_pairs": sum(row["candidate_count"] for row in manifests),
            "queries_by_split": dict(
                sorted(Counter(row["internal_family_split"] for row in manifests).items())
            ),
            "queries_by_source_type": dict(
                sorted(Counter(row["query_source_type"] for row in manifests).items())
            ),
            "source_anchor_needed": sum(
                row["audit_source_anchor_needed"] for row in manifests
            ),
        },
        "payload_estimate": {
            "total_prompt_chars_excluding_system_prompt": prompt_chars,
            "rough_input_tokens_chars_div_4": prompt_chars // 4,
            "pilot_observed_total_tokens_per_query": 210163 / 30,
            "projected_total_tokens_from_pilot": round(
                (210163 / 30) * len(manifests)
            ),
        },
        "judge_contract": {
            "model": "MiniMax-M2.7",
            "one_query_one_request": True,
            "candidates_per_query": 20,
            "grades": [0, 1, 2, "uncertain"],
            "numeric_confidence_forbidden": True,
            "official_eval_tasks_forbidden": True,
            "explicit_user_api_authorization_received": True,
        },
        "private_manifest": str(MANIFEST.relative_to(ROOT)),
    }
    dump_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
