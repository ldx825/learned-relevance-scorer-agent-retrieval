#!/usr/bin/env python3
"""Build conservative, retrieval-aligned SkillsBench 2/1/0 training pairs.

This stage makes no API calls and never upgrades a label because a skill was
the query's construction source.  It keeps only independently judged pairs
that came from one of the three retrieval branches used by ALFWorld V3.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from source_policy import ROOT, assert_training_source, load_policy


LABELS = ROOT / "data/skillbench_ncf/graded_judge_full_v4/final_labels.jsonl"
CANDIDATES = ROOT / "data/skillbench_ncf/candidate_pool_v4_balanced/candidates.jsonl"
QUERIES = ROOT / "data/skillbench_ncf/expanded_queries_v2/queries.jsonl"
OUTPUT_DIR = ROOT / "data/skillbench_ncf/graded_judge_full_v4_calibrated"
REPORT = ROOT / "artifacts/skillbench_ncf/manifests/graded_judge_full_v4_calibration_report.json"
RETRIEVAL_SOURCES = {
    "base_cosine",
    "hybrid_cosine",
    "v2_ncf_balanced_hard",
}
VALID_GRADES = {0, 1, 2, "uncertain"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    policy = load_policy()
    labels_path = assert_training_source(LABELS, policy)
    candidates_path = assert_training_source(CANDIDATES, policy)
    queries_path = assert_training_source(QUERIES, policy)

    labels = load_jsonl(labels_path)
    candidates = {row["pair_id"]: row for row in load_jsonl(candidates_path)}
    queries = {row["query_id"]: row for row in load_jsonl(queries_path)}
    if len(labels) != 73_000 or len(queries) != 3_650:
        raise AssertionError("locked V4 Judge corpus changed unexpectedly")
    if len({row["pair_id"] for row in labels}) != len(labels):
        raise AssertionError("duplicate Judge pair IDs")
    if any(row.get("official_skillsbench_task_used") for row in labels):
        raise AssertionError("official SkillsBench task flag reached Judge labels")

    prefiltered: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = Counter()
    raw_grades = Counter()
    retained_actions = Counter()
    for label in labels:
        pair_id = str(label["pair_id"])
        candidate = candidates.get(pair_id)
        if candidate is None:
            raise KeyError(f"Judge pair missing from candidate pool: {pair_id}")
        query = queries.get(str(label["query_id"]))
        if query is None:
            raise KeyError(f"Judge query missing from query corpus: {label['query_id']}")
        if candidate["query_id"] != label["query_id"]:
            raise AssertionError(f"query mismatch: {pair_id}")
        if candidate["candidate_skill_id"] != label["candidate_skill_id"]:
            raise AssertionError(f"candidate mismatch: {pair_id}")
        if candidate["internal_family_split"] != query["internal_family_split"]:
            raise AssertionError(f"split mismatch: {pair_id}")

        grade = label["raw_judge_grade"]
        if grade not in VALID_GRADES:
            raise ValueError(f"invalid Judge grade {grade!r}: {pair_id}")
        raw_grades[str(grade)] += 1
        if grade == "uncertain":
            excluded["uncertain"] += 1
            continue
        sources = set(candidate["candidate_sources"])
        if not sources & RETRIEVAL_SOURCES:
            # ALFWorld V3 likewise excludes graph-only candidates.  SkillsBench
            # source anchors and easy probes are construction diagnostics, not
            # runtime retrieval branches, so they are excluded by the same rule.
            excluded["no_runtime_retrieval_source"] += 1
            continue
        if int(grade) > 0 and not str(label.get("judge_evidence", "")).strip():
            excluded["positive_without_judge_evidence"] += 1
            continue

        action = "retain_independent_judge_grade"
        retained_actions[action] += 1
        prefiltered[str(label["query_id"])].append(
            {
                "schema_version": "skillbench_ncf.calibrated_pair.v4",
                "pair_id": pair_id,
                "query_id": label["query_id"],
                "query_text": candidate["query_text"],
                "query_source_type": candidate["query_source_type"],
                "internal_family_split": candidate["internal_family_split"],
                "source_skill_id": candidate["source_skill_id"],
                "candidate_skill_id": candidate["candidate_skill_id"],
                "candidate_family_id": candidate["candidate_family_id"],
                "candidate_sources": candidate["candidate_sources"],
                "known_source_pair": candidate["known_source_pair"],
                "raw_judge_grade": grade,
                "target_grade": int(grade),
                "target_relevance": int(grade) / 2.0,
                "calibration_action": action,
                "judge_evidence": label.get("judge_evidence", ""),
                "judge_reason_zh": label.get("judge_reason_zh", ""),
                "normalization_action": label.get("normalization_action", "none"),
                "label_source": "minimax_m27_independent_candidate_judge_v4",
                "training_ready": True,
                "official_skillsbench_task_used": False,
            }
        )

    # A ranking group without a direct grade-2 item cannot teach required-skill
    # retrieval.  Drop the entire group rather than silently treating grade 1 as
    # the best available answer.
    no_grade2_queries = {
        query_id
        for query_id, rows in prefiltered.items()
        if not any(row["target_grade"] == 2 for row in rows)
    }
    rows = [
        row
        for query_id, group in prefiltered.items()
        if query_id not in no_grade2_queries
        for row in group
    ]
    excluded["pairs_in_query_without_retained_grade2"] = sum(
        len(prefiltered[query_id]) for query_id in no_grade2_queries
    )
    rows.sort(
        key=lambda row: (
            row["internal_family_split"],
            row["query_id"],
            row["candidate_skill_id"],
        )
    )
    output = OUTPUT_DIR / "pairs.jsonl"
    dump_jsonl(output, rows)

    report = {
        "schema_version": "skillbench_ncf.calibrated_pairs_report.v4",
        "status": "calibration_complete_training_ready",
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "calibration_contract": {
            "source_skill_automatically_forced_to_grade2": False,
            "semantic_grade_upgrades_or_downgrades": False,
            "uncertain_enters_training": False,
            "query_without_retained_grade2_enters_training": False,
            "runtime_retrieval_sources": sorted(RETRIEVAL_SOURCES),
            "graph_only_source_anchor_only_easy_probe_enter_training": False,
            "positive_requires_nonempty_judge_evidence": True,
        },
        "counts": {
            "input_labels": len(labels),
            "input_queries": len(queries),
            "retained_pairs": len(rows),
            "retained_queries": len({row["query_id"] for row in rows}),
            "retained_pairs_by_split": dict(
                sorted(Counter(row["internal_family_split"] for row in rows).items())
            ),
            "retained_queries_by_split": {
                split: len(
                    {
                        row["query_id"]
                        for row in rows
                        if row["internal_family_split"] == split
                    }
                )
                for split in ("train", "dev")
            },
            "raw_grade_counts": dict(sorted(raw_grades.items())),
            "target_grade_counts": dict(
                sorted(Counter(str(row["target_grade"]) for row in rows).items())
            ),
            "excluded": dict(sorted(excluded.items())),
            "queries_without_retained_grade2": len(no_grade2_queries),
            "calibration_actions": dict(sorted(retained_actions.items())),
        },
        "provenance": {
            "labels_sha256": sha256_file(labels_path),
            "candidates_sha256": sha256_file(candidates_path),
            "queries_sha256": sha256_file(queries_path),
            "private_pairs": str(output.relative_to(ROOT)),
            "private_pairs_sha256": sha256_file(output),
        },
    }
    dump_json(OUTPUT_DIR / "report.json", report)
    dump_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
