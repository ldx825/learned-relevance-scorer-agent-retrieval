#!/usr/bin/env python3
"""Build a fixed, zero-API 30-query pilot for graded candidate judging."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_evidence_v1 import ROOT, dump_json, dump_jsonl, load_jsonl
from source_policy import assert_training_source, load_policy


CANDIDATES = (
    ROOT / "data/skillbench_ncf/candidate_pool_v4_balanced/candidates.jsonl"
)
EVIDENCE = ROOT / "data/skillbench_ncf/evidence_v1/skill_evidence_cards.jsonl"
ENRICHED = (
    ROOT
    / "data/skillbench_ncf/enriched_skill_embeddings_v3/representation_texts.jsonl"
)
OUTPUT_DIR = ROOT / "data/skillbench_ncf/graded_judge_pilot_v4"
MANIFEST = OUTPUT_DIR / "judge_manifest.jsonl"
REPORT = (
    ROOT
    / "artifacts/skillbench_ncf/manifests/graded_judge_pilot_v4_report.json"
)
REVIEW = (
    ROOT
    / "artifacts/skillbench_ncf/manifests/graded_judge_pilot_v4_review.json"
)

MAX_CANDIDATES = 20
LOCAL_OPERATION_LIMIT = 4
PUBLIC_OPERATION_LIMIT = 5

# 24 train + 6 held-out-family dev. The mix reflects the source corpus while
# guaranteeing that every query construction route is inspected.
QUOTAS = {
    ("linked_public_composio_suggested_prompt", "train"): 10,
    ("linked_public_composio_suggested_prompt", "dev"): 2,
    ("locked_skill_md_tool_discovery_anchor", "train"): 5,
    ("locked_skill_md_tool_discovery_anchor", "dev"): 1,
    ("locked_skill_md_operation", "train"): 4,
    ("locked_skill_md_operation", "dev"): 2,
    ("locked_skill_md_description", "train"): 4,
    ("locked_skill_md_description", "dev"): 1,
    ("locked_skill_md_dedup_coverage_fallback", "train"): 1,
}


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())
        if token
        not in {
            "this",
            "that",
            "with",
            "from",
            "into",
            "your",
            "using",
            "skill",
            "complete",
            "documented",
            "operation",
        }
    }


def relevance(text: str, query_tokens: set[str]) -> tuple[int, int, str]:
    compact = re.sub(r"\s+", " ", text).strip()
    overlap = len(tokens(compact) & query_tokens)
    return (-overlap, len(compact), compact)


def evidence_summary(
    skill_id: str,
    query_text: str,
    cards: dict[str, dict[str, Any]],
    enriched: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    card = cards[skill_id]
    query_tokens = tokens(query_text)
    operations = sorted(
        card.get("operation_cards", []),
        key=lambda row: relevance(
            f"{row.get('title', '')} {row.get('summary', '')}",
            query_tokens,
        ),
    )[:LOCAL_OPERATION_LIMIT]
    local_evidence = [
        re.sub(
            r"\s+",
            " ",
            f"{row.get('title', '')}: {row.get('summary', '')}",
        ).strip()[:600]
        for row in operations
    ]

    public_evidence: list[str] = []
    representation = enriched.get(skill_id, {}).get("representation_text")
    if representation:
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in str(representation).splitlines()
            if line.lstrip().startswith("- ")
        ]
        public_evidence = sorted(
            lines, key=lambda line: relevance(line, query_tokens)
        )[:PUBLIC_OPERATION_LIMIT]
        public_evidence = [line[:600] for line in public_evidence]

    return {
        "skill_id": skill_id,
        "description": str(card["description"])[:700],
        "local_operation_evidence": local_evidence,
        "linked_public_operation_evidence": public_evidence,
        "evidence_class": card["evidence_class"],
    }


def candidate_priority(row: dict[str, Any]) -> tuple[int, int, str]:
    sources = set(row["candidate_sources"])
    if row["known_source_pair"]:
        group = 0
    elif sources & {"base_cosine", "hybrid_cosine"}:
        group = 1
    elif "v2_ncf_balanced_hard" in sources:
        group = 2
    elif "graph_one_hop" in sources:
        group = 3
    else:
        group = 4
    ranks = [
        int(row[key])
        for key in (
            "base_cosine_rank",
            "hybrid_cosine_rank",
            "v2_ncf_balanced_hard_rank",
            "deterministic_easy_probe_rank",
        )
        if row.get(key) is not None
    ]
    return (group, min(ranks, default=10_000), str(row["candidate_skill_id"]))


def choose_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the source anchor, retrieval leaders, and diagnostic hard/easy rows."""

    selected: dict[str, dict[str, Any]] = {}

    # Source provenance is retained only for auditing. It is not identified as
    # the answer in the payload sent to the Judge.
    for row in rows:
        if row["known_source_pair"]:
            selected[str(row["candidate_skill_id"])] = row

    # Preserve two diagnostic examples from each non-cosine branch.
    for source in (
        "v2_ncf_balanced_hard",
        "graph_one_hop",
        "deterministic_easy_probe",
    ):
        branch = [row for row in rows if source in row["candidate_sources"]]
        for row in sorted(branch, key=candidate_priority)[:2]:
            selected[str(row["candidate_skill_id"])] = row

    # Fill with the strongest union of the two semantic retrieval views.
    for row in sorted(rows, key=candidate_priority):
        if len(selected) >= MAX_CANDIDATES:
            break
        selected.setdefault(str(row["candidate_skill_id"]), row)
    return sorted(selected.values(), key=candidate_priority)


def main() -> int:
    policy = load_policy()
    for path in (CANDIDATES, EVIDENCE, ENRICHED):
        assert_training_source(path, policy)

    candidate_rows = load_jsonl(CANDIDATES)
    cards = {row["skill_id"]: row for row in load_jsonl(EVIDENCE)}
    enriched = {row["skill_id"]: row for row in load_jsonl(ENRICHED)}
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_query[str(row["query_id"])].append(row)

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for query_id, rows in by_query.items():
        first = rows[0]
        buckets[
            (
                str(first["query_source_type"]),
                str(first["internal_family_split"]),
            )
        ].append(
            {
                "query_id": query_id,
                "rows": rows,
                "source_anchor_needed": any(
                    "source_anchor" in row["candidate_sources"] for row in rows
                ),
                "has_graph_candidate": any(
                    "graph_one_hop" in row["candidate_sources"] for row in rows
                ),
            }
        )

    chosen: list[dict[str, Any]] = []
    for bucket, quota in QUOTAS.items():
        available = buckets[bucket]
        # Interleave anchor/retrieved cases and prefer graph-bearing examples,
        # then use a stable hash rather than human-picked task semantics.
        ordered = sorted(
            available,
            key=lambda item: (
                int(not item["has_graph_candidate"]),
                stable_key(item["query_id"]),
            ),
        )
        anchor = [item for item in ordered if item["source_anchor_needed"]]
        retrieved = [item for item in ordered if not item["source_anchor_needed"]]
        interleaved = []
        while anchor or retrieved:
            if retrieved:
                interleaved.append(retrieved.pop(0))
            if anchor:
                interleaved.append(anchor.pop(0))
        if len(interleaved) < quota:
            raise ValueError(f"not enough queries for {bucket}: {len(interleaved)}")
        chosen.extend(interleaved[:quota])

    manifests = []
    for item in sorted(chosen, key=lambda row: row["query_id"]):
        rows = item["rows"]
        first = rows[0]
        candidates = choose_candidates(rows)
        manifests.append(
            {
                "schema_version": "skillbench_ncf.graded_judge_query.v4",
                "query_id": first["query_id"],
                "query_text": first["query_text"],
                "query_source_type": first["query_source_type"],
                "internal_family_split": first["internal_family_split"],
                "audit_source_skill_id": first["source_skill_id"],
                "audit_source_anchor_needed": item["source_anchor_needed"],
                "official_skillsbench_task_used": False,
                "candidate_count": len(candidates),
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
                    for row in candidates
                ],
                "labels": None,
                "api_submission_authorized": False,
            }
        )

    if len(manifests) != 30 or len({row["query_id"] for row in manifests}) != 30:
        raise AssertionError("pilot must contain exactly 30 unique queries")
    if any(row["labels"] is not None for row in manifests):
        raise AssertionError("zero-API manifest unexpectedly contains labels")
    if any(row["official_skillsbench_task_used"] for row in manifests):
        raise AssertionError("official SkillsBench task leakage")

    dump_jsonl(MANIFEST, manifests)
    stats = {
        "schema_version": "skillbench_ncf.graded_judge_pilot_report.v4",
        "status": "manifest_complete_zero_api",
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "selection_is_hash_stable_not_result_selected": True,
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
            "queries_with_graph_candidates": sum(
                any(
                    "graph_one_hop" in candidate["candidate_sources"]
                    for candidate in row["candidates"]
                )
                for row in manifests
            ),
        },
        "candidate_count": {
            "min": min(row["candidate_count"] for row in manifests),
            "max": max(row["candidate_count"] for row in manifests),
            "mean": sum(row["candidate_count"] for row in manifests)
            / len(manifests),
        },
        "judge_contract": {
            "model": "MiniMax-M2.7",
            "one_query_one_candidate_set": True,
            "grades": [0, 1, 2, "uncertain"],
            "numeric_confidence_forbidden": True,
            "source_answer_hidden_from_api_prompt": True,
            "same_model_bias_remains_a_limitation": True,
            "api_submission_requires_explicit_user_authorization": True,
        },
        "private_manifest": str(MANIFEST.relative_to(ROOT)),
    }
    dump_json(REPORT, stats)
    dump_json(
        REVIEW,
        {
            "warning": "This review file contains no Judge labels.",
            "queries": manifests[:5],
        },
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
