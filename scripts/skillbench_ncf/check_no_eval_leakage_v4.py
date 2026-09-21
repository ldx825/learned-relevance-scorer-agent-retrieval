#!/usr/bin/env python3
"""Fail-closed provenance, split, and leakage audit for SkillsBench V4 pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from source_policy import ROOT, assert_training_source, load_policy


PAIRS = ROOT / "data/skillbench_ncf/graded_judge_full_v4_calibrated/pairs.jsonl"
QUERIES = ROOT / "data/skillbench_ncf/expanded_queries_v2/queries.jsonl"
FAMILIES = ROOT / "artifacts/skillbench_ncf/manifests/skill_family_manifest.jsonl"
REPORT = ROOT / "artifacts/skillbench_ncf/manifests/graded_judge_full_v4_leakage_audit.json"
RETRIEVAL_SOURCES = {
    "base_cosine",
    "hybrid_cosine",
    "v2_ncf_balanced_hard",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_hash(value: str) -> str:
    normalized = " ".join(value.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> int:
    policy = load_policy()
    pairs_path = assert_training_source(PAIRS, policy)
    queries_path = assert_training_source(QUERIES, policy)
    # The committed family manifest is an audit artifact, not task content.
    families_path = FAMILIES.resolve()
    pairs = load_jsonl(pairs_path)
    queries = {row["query_id"]: row for row in load_jsonl(queries_path)}
    families = {row["skill_id"]: row for row in load_jsonl(families_path)}

    if len(families) != 1_000 or set(row["split"] for row in families.values()) != {
        "train",
        "dev",
    }:
        raise AssertionError("family manifest is not the locked 1000-skill split")
    pair_ids = [row["pair_id"] for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("duplicate calibrated pair IDs")
    if any(row.get("official_skillsbench_task_used") for row in pairs):
        raise AssertionError("official SkillsBench task flag reached calibrated pairs")

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        query = queries.get(row["query_id"])
        if query is None:
            raise KeyError(f"unknown query: {row['query_id']}")
        if row["query_text"] != query["query_text"]:
            raise AssertionError(f"query text changed: {row['query_id']}")
        split = row["internal_family_split"]
        if split != query["internal_family_split"]:
            raise AssertionError(f"query split changed: {row['query_id']}")
        if families[row["source_skill_id"]]["split"] != split:
            raise AssertionError(f"source family crosses split: {row['pair_id']}")
        if families[row["candidate_skill_id"]]["split"] != split:
            raise AssertionError(f"candidate family crosses split: {row['pair_id']}")
        if row["candidate_family_id"] != families[row["candidate_skill_id"]]["family_id"]:
            raise AssertionError(f"candidate family ID mismatch: {row['pair_id']}")
        if not set(row["candidate_sources"]) & RETRIEVAL_SOURCES:
            raise AssertionError(f"non-runtime candidate survived: {row['pair_id']}")
        if row["target_grade"] not in {0, 1, 2}:
            raise AssertionError(f"invalid target grade: {row['pair_id']}")
        if not row["training_ready"]:
            raise AssertionError(f"non-ready row survived: {row['pair_id']}")
        by_query[row["query_id"]].append(row)

    if any(not any(row["target_grade"] == 2 for row in rows) for rows in by_query.values()):
        raise AssertionError("a retained query lacks grade 2")

    text_splits: dict[str, set[str]] = defaultdict(set)
    for query_id in by_query:
        query = queries[query_id]
        text_splits[normalized_text_hash(query["query_text"])].add(
            query["internal_family_split"]
        )
    cross_split_texts = [value for value, splits in text_splits.items() if len(splits) > 1]
    if cross_split_texts:
        raise AssertionError(
            f"normalized query text crosses train/dev: {len(cross_split_texts)}"
        )

    train_families = {
        families[row["source_skill_id"]]["family_id"]
        for row in pairs
        if row["internal_family_split"] == "train"
    } | {
        families[row["candidate_skill_id"]]["family_id"]
        for row in pairs
        if row["internal_family_split"] == "train"
    }
    dev_families = {
        families[row["source_skill_id"]]["family_id"]
        for row in pairs
        if row["internal_family_split"] == "dev"
    } | {
        families[row["candidate_skill_id"]]["family_id"]
        for row in pairs
        if row["internal_family_split"] == "dev"
    }
    if train_families & dev_families:
        raise AssertionError("skill family crosses train/dev")

    result = {
        "schema_version": "skillbench_ncf.leakage_audit.v4",
        "status": "pass",
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "audit_scope": {
            "source_boundary": (
                "Training builders are path-allowlisted and official task roots are "
                "forbidden. The audit intentionally does not read official evaluation "
                "tasks merely to compare their text."
            ),
            "content_overlap_checked": "exact normalized train/dev query text only",
            "family_isolation_checked": True,
            "privileged_fields_used_as_model_inputs": False,
        },
        "counts": {
            "pairs": len(pairs),
            "queries": len(by_query),
            "pairs_by_split": dict(
                sorted(Counter(row["internal_family_split"] for row in pairs).items())
            ),
            "queries_by_split": {
                split: sum(
                    queries[query_id]["internal_family_split"] == split
                    for query_id in by_query
                )
                for split in ("train", "dev")
            },
            "train_families": len(train_families),
            "dev_families": len(dev_families),
            "cross_split_families": 0,
            "cross_split_normalized_query_texts": 0,
        },
        "provenance": {
            "pairs_sha256": sha256_file(pairs_path),
            "queries_sha256": sha256_file(queries_path),
            "family_manifest_sha256": sha256_file(families_path),
            "source_policy_sha256": sha256_file(
                ROOT / "configs/skillbench_ncf/eval_isolation.json"
            ),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, ValueError, PermissionError) as exc:
        print(f"[FAILED] SkillsBench V4 leakage audit: {exc}")
        raise SystemExit(1)
