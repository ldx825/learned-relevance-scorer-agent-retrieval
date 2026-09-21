#!/usr/bin/env python3
"""Fail closed when the V4 unlabeled candidate contract is violated."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / "data/skillbench_ncf/candidate_pool_v4/candidates.jsonl"
EVIDENCE = ROOT / "data/skillbench_ncf/evidence_v1/skill_evidence_cards.jsonl"
REPORT = ROOT / "artifacts/skillbench_ncf/manifests/candidate_pool_v4_report.json"
ALLOWED_SOURCES = {
    "base_cosine",
    "hybrid_cosine",
    "v2_ncf_hard",
    "graph_one_hop",
    "deterministic_easy_probe",
    "source_anchor",
}


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    pairs = rows(CANDIDATES)
    cards = {row["skill_id"]: row for row in rows(EVIDENCE)}
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if len(pairs) != report["counts"]["candidate_pairs"]:
        raise AssertionError("candidate count differs from public report")
    pair_ids = [row["pair_id"] for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("duplicate query-skill pair")
    grouped = defaultdict(list)
    for row in pairs:
        grouped[row["query_id"]].append(row)
        if row["label"] is not None or not row["candidate_only"]:
            raise AssertionError("unreviewed candidate received a label")
        if row.get("official_skillsbench_task_used"):
            raise AssertionError("official evaluation content flag is true")
        if not set(row["candidate_sources"]) <= ALLOWED_SOURCES:
            raise AssertionError(f"unknown candidate source: {row['pair_id']}")
        if cards[row["candidate_skill_id"]]["split"] != row["internal_family_split"]:
            raise AssertionError(f"cross-split skill exposure: {row['pair_id']}")
        if "source_anchor" in row["candidate_sources"] and not row["known_source_pair"]:
            raise AssertionError(f"source anchor applied to non-source: {row['pair_id']}")
    if len(grouped) != report["counts"]["queries"]:
        raise AssertionError("query count differs from public report")
    for query_id, candidates in grouped.items():
        if sum(row["known_source_pair"] for row in candidates) != 1:
            raise AssertionError(f"expected one provenance source pair: {query_id}")
    if set(cards) != {row["candidate_skill_id"] for row in pairs}:
        raise AssertionError("candidate pool does not expose all 1000 skills")
    source_counts = Counter(
        source for row in pairs for source in row["candidate_sources"]
    )
    if dict(sorted(source_counts.items())) != report["counts"]["candidate_pairs_by_source"]:
        raise AssertionError("candidate source counts differ from public report")
    if report["label_contract"]["judge_ready"]:
        raise AssertionError("biased raw pool must not be marked judge-ready")
    print("[PASS] V4 pool is split-safe, unlabeled, complete, and not judge-ready")
    print(f"queries={len(grouped)} pairs={len(pairs)} skills={len(cards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
