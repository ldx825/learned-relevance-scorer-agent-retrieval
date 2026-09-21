#!/usr/bin/env python3
"""Create V4.1 arrays with evidence-aware and monotonic grade weighting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from source_policy import ROOT, assert_training_source, load_policy


PAIRS = ROOT / "data/skillbench_ncf/graded_judge_full_v4_calibrated/pairs.jsonl"
V4_ARRAYS = ROOT / "data/skillbench_ncf/model_data/neumf_full_v4/arrays.npz"
OUTPUT = ROOT / "data/skillbench_ncf/model_data/neumf_full_v4_1"
REPORT = ROOT / "artifacts/skillbench_ncf/manifests/neumf_full_v4_1_weight_report.json"
SOURCE_SPECIFICITY = {
    "linked_public_composio_suggested_prompt": 1.0,
    "locked_skill_md_operation": 1.0,
    "locked_skill_md_description": 0.5,
    "locked_skill_md_tool_discovery_anchor": 0.2,
    "locked_skill_md_dedup_coverage_fallback": 0.05,
}
GRADE_FACTOR = {0: 1.0, 1: 2.0, 2: 3.0}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    policy = load_policy()
    pairs_path = assert_training_source(PAIRS, policy)
    arrays_path = assert_training_source(V4_ARRAYS, policy)
    pairs = load_jsonl(pairs_path)
    with np.load(arrays_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    if len(pairs) != len(arrays["pair_ids"]) or len(pairs) != 51_911:
        raise AssertionError("V4 pair/array alignment changed")
    if [row["pair_id"] for row in pairs] != arrays["pair_ids"].astype(str).tolist():
        raise AssertionError("V4 pair order differs from arrays")

    split = arrays["split_codes"]
    grade = arrays["target_grade"]
    train_offsets = np.flatnonzero(split == 0)
    query_count = Counter(pairs[i]["query_id"] for i in train_offsets)
    skill_count = Counter(pairs[i]["candidate_skill_id"] for i in train_offsets)
    train_count = len(train_offsets)
    raw_weights = []
    for offset in train_offsets:
        row = pairs[int(offset)]
        source_factor = SOURCE_SPECIFICITY[row["query_source_type"]]
        raw_weights.append(
            1.0
            / query_count[row["query_id"]]
            * source_factor
            * GRADE_FACTOR[int(grade[offset])]
            * np.sqrt(train_count / skill_count[row["candidate_skill_id"]])
        )
    raw = np.asarray(raw_weights, dtype=np.float64)
    p99 = float(np.quantile(raw, 0.99))
    clipped = np.minimum(raw, p99)
    normalized = clipped / clipped.mean()
    sample_weight = np.ones(len(pairs), dtype=np.float32)
    sample_weight[train_offsets] = normalized.astype(np.float32)
    arrays["sample_weight"] = sample_weight

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_arrays = OUTPUT / "arrays.npz"
    np.savez_compressed(output_arrays, **arrays)

    mass_by_source = defaultdict(float)
    mass_by_grade = defaultdict(float)
    pairs_by_source = Counter()
    for offset in train_offsets:
        row = pairs[int(offset)]
        source = row["query_source_type"]
        pairs_by_source[source] += 1
        mass_by_source[source] += float(sample_weight[offset])
        mass_by_grade[str(int(grade[offset]))] += float(sample_weight[offset])
    total_mass = float(sample_weight[train_offsets].sum())
    report = {
        "schema_version": "skillbench_ncf.neumf_full_v4_1_weights.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "status": "ready",
        "changes_from_v4": [
            "removed inverse query-source frequency equalization",
            "added evidence-specific query-source factors",
            "added monotonic grade factors 0:1, 1:2, 2:3",
            "retained per-query normalization and inverse-sqrt skill exposure",
            "retained train-only P99 clipping and mean-one normalization",
        ],
        "source_specificity": SOURCE_SPECIFICITY,
        "grade_factor": GRADE_FACTOR,
        "counts": {
            "pairs": len(pairs),
            "train_pairs": len(train_offsets),
            "dev_pairs": int((split == 1).sum()),
            "pairs_by_train_source": dict(sorted(pairs_by_source.items())),
        },
        "train_weight": {
            "raw_p99": p99,
            "min": float(sample_weight[train_offsets].min()),
            "max": float(sample_weight[train_offsets].max()),
            "mean": float(sample_weight[train_offsets].mean()),
            "mass_share_by_source": {
                key: value / total_mass
                for key, value in sorted(mass_by_source.items())
            },
            "mass_share_by_grade": {
                key: value / total_mass
                for key, value in sorted(mass_by_grade.items())
            },
        },
        "provenance": {
            "v4_arrays_sha256": sha256_file(arrays_path),
            "pairs_sha256": sha256_file(pairs_path),
            "private_arrays": str(output_arrays.relative_to(ROOT)),
            "private_arrays_sha256": sha256_file(output_arrays),
        },
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
