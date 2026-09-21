#!/usr/bin/env python3
"""Assemble content-only NeuMF arrays from calibrated SkillsBench V4 pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from source_policy import ROOT, assert_training_source, load_policy


PAIRS = ROOT / "data/skillbench_ncf/graded_judge_full_v4_calibrated/pairs.jsonl"
QUERY_EMBEDDINGS = ROOT / "data/skillbench_ncf/expanded_pairs_v2_balanced/query_embeddings.json"
SKILL_EMBEDDINGS = ROOT / "data/skillbench_ncf/enriched_skill_embeddings_v3/hybrid_skill_embeddings.json"
OUTPUT = ROOT / "data/skillbench_ncf/model_data/neumf_full_v4"
PUBLIC_REPORT = ROOT / "artifacts/skillbench_ncf/manifests/neumf_full_v4_model_data_report.json"
SPLIT_CODES = {"train": 0, "dev": 1}
SOURCE_TYPES = (
    "linked_public_composio_suggested_prompt",
    "locked_skill_md_description",
    "locked_skill_md_operation",
    "locked_skill_md_tool_discovery_anchor",
    "locked_skill_md_dedup_coverage_fallback",
)
RETRIEVAL_SOURCES = (
    "base_cosine",
    "hybrid_cosine",
    "v2_ncf_balanced_hard",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_items(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["model"]), payload.get("items", payload)


def main() -> int:
    policy = load_policy()
    pairs_path = assert_training_source(PAIRS, policy)
    query_embeddings_path = assert_training_source(QUERY_EMBEDDINGS, policy)
    skill_embeddings_path = assert_training_source(SKILL_EMBEDDINGS, policy)
    pairs = load_jsonl(pairs_path)
    if not pairs:
        raise AssertionError("no calibrated pairs")

    query_model, query_cache = embedding_items(query_embeddings_path)
    skill_model, skill_cache = embedding_items(skill_embeddings_path)
    if {query_model, skill_model} != {"text-embedding-3-large"}:
        raise AssertionError("query and skill representations must use text-embedding-3-large")
    query_ids = sorted({str(row["query_id"]) for row in pairs})
    skill_ids = sorted(skill_cache)
    if len(skill_ids) != 1_000:
        raise AssertionError("expected exactly 1000 skill representations")
    if not set(query_ids) <= set(query_cache):
        raise AssertionError("query embedding cache is incomplete")

    query_index = {value: index for index, value in enumerate(query_ids)}
    skill_index = {value: index for index, value in enumerate(skill_ids)}
    query_embeddings = np.asarray(
        [query_cache[value]["embedding"] for value in query_ids], dtype=np.float32
    )
    skill_embeddings = np.asarray(
        [skill_cache[value]["embedding"] for value in skill_ids], dtype=np.float32
    )
    if query_embeddings.shape[1:] != (3072,) or skill_embeddings.shape != (1000, 3072):
        raise AssertionError(
            f"unexpected embedding shapes: {query_embeddings.shape}, {skill_embeddings.shape}"
        )

    train_pairs = [row for row in pairs if row["internal_family_split"] == "train"]
    query_frequency = Counter(row["query_id"] for row in train_pairs)
    source_frequency = Counter(row["query_source_type"] for row in train_pairs)
    skill_frequency = Counter(row["candidate_skill_id"] for row in train_pairs)
    train_count = len(train_pairs)
    active_source_types = tuple(sorted(source_frequency))
    source_factor = {
        source: train_count / (len(active_source_types) * count)
        for source, count in source_frequency.items()
    }
    skill_factor = {
        skill_id: np.sqrt(train_count / count)
        for skill_id, count in skill_frequency.items()
    }

    pair_query_indices = np.asarray(
        [query_index[row["query_id"]] for row in pairs], dtype=np.int32
    )
    pair_skill_indices = np.asarray(
        [skill_index[row["candidate_skill_id"]] for row in pairs], dtype=np.int16
    )
    target_grade = np.asarray([row["target_grade"] for row in pairs], dtype=np.int8)
    target_relevance = target_grade.astype(np.float32) / 2.0
    split_codes = np.asarray(
        [SPLIT_CODES[row["internal_family_split"]] for row in pairs], dtype=np.int8
    )
    source_type_codes = np.asarray(
        [SOURCE_TYPES.index(row["query_source_type"]) for row in pairs], dtype=np.int8
    )
    candidate_source_features = np.asarray(
        [
            [float(source in row["candidate_sources"]) for source in RETRIEVAL_SOURCES]
            for row in pairs
        ],
        dtype=np.float32,
    )

    sample_weight = np.ones(len(pairs), dtype=np.float32)
    raw_weights: list[float] = []
    train_offsets: list[int] = []
    for offset, row in enumerate(pairs):
        if row["internal_family_split"] != "train":
            continue
        weight = (
            1.0
            / query_frequency[row["query_id"]]
            * source_factor[row["query_source_type"]]
            * skill_factor[row["candidate_skill_id"]]
        )
        raw_weights.append(float(weight))
        train_offsets.append(offset)
    # Rare source-type × rare-skill combinations can otherwise receive
    # two-hundred-fold weight and dominate a batch.  Determine the cap from
    # train only, clip at its P99, then renormalize to mean one.  This is a
    # fixed, label-agnostic stability rule rather than a dev-tuned knob.
    raw_weight_array = np.asarray(raw_weights, dtype=np.float64)
    train_weight_p99 = float(np.quantile(raw_weight_array, 0.99))
    clipped_weights = np.minimum(raw_weight_array, train_weight_p99)
    normalizer = float(np.mean(clipped_weights))
    for offset, weight in zip(train_offsets, clipped_weights, strict=True):
        sample_weight[offset] = float(weight / normalizer)

    query_norm = query_embeddings / np.maximum(
        np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12
    )
    skill_norm = skill_embeddings / np.maximum(
        np.linalg.norm(skill_embeddings, axis=1, keepdims=True), 1e-12
    )
    cosine = np.sum(
        query_norm[pair_query_indices] * skill_norm[pair_skill_indices], axis=1
    ).astype(np.float32)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays_path = OUTPUT / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        task_ids=np.asarray(query_ids),
        skill_ids=np.asarray(skill_ids),
        pair_ids=np.asarray([row["pair_id"] for row in pairs]),
        task_embeddings=query_embeddings,
        # Compatibility aliases let SkillsBench reuse the exact ALFWorld V3
        # pointwise+pairwise training implementation without changing its math.
        phase_embeddings=query_embeddings,
        skill_embeddings=skill_embeddings,
        pair_task_indices=pair_query_indices,
        pair_phase_indices=pair_query_indices,
        pair_skill_indices=pair_skill_indices,
        pair_features=cosine[:, None],
        target_grade=target_grade,
        target_relevance=target_relevance,
        sample_weight=sample_weight,
        split_codes=split_codes,
        query_source_type_codes=source_type_codes,
        candidate_source_features=candidate_source_features,
    )

    grade_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pairs:
        grade_by_split[row["internal_family_split"]][str(row["target_grade"])] += 1
    train_weights = sample_weight[split_codes == SPLIT_CODES["train"]]
    grade2_skill_counts = Counter(
        row["candidate_skill_id"]
        for row in train_pairs
        if row["target_grade"] == 2
    )
    total_train_grade2 = sum(grade2_skill_counts.values())
    top10_grade2 = sum(count for _, count in grade2_skill_counts.most_common(10))
    metadata = {
        "schema_version": "skillbench_ncf.content_neumf_model_data.v4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "training_arrays_ready",
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "embedding_model": query_model,
        "embedding_dimension": 3072,
        "counts": {
            "queries": len(query_ids),
            "skills_in_catalog": len(skill_ids),
            "pairs": len(pairs),
            "pairs_by_split": dict(
                sorted(Counter(row["internal_family_split"] for row in pairs).items())
            ),
            "queries_by_split": {
                split: len(
                    {
                        row["query_id"]
                        for row in pairs
                        if row["internal_family_split"] == split
                    }
                )
                for split in SPLIT_CODES
            },
            "candidate_skills_by_split": {
                split: len(
                    {
                        row["candidate_skill_id"]
                        for row in pairs
                        if row["internal_family_split"] == split
                    }
                )
                for split in SPLIT_CODES
            },
            "grade_counts_by_split": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(grade_by_split.items())
            },
        },
        "content_only_contract": {
            "learned_query_id_embedding": False,
            "learned_skill_id_embedding": False,
            "model_inputs": [
                "query_text_embedding",
                "hybrid_skill_document_and_public_tool_embedding",
            ],
            "ids_used_only_as_array_indices": True,
            "candidate_source_features_used_by_model": False,
        },
        "training_weight_contract": {
            "statistics_source": "internal train split only",
            "formula": (
                "1/query_pair_count * inverse_query_source_frequency "
                "* inverse_sqrt_candidate_skill_frequency; train P99 clip"
            ),
            "outlier_clip": "train-only raw-weight P99 before mean normalization",
            "raw_train_p99": train_weight_p99,
            "normalized_train_mean": float(train_weights.mean()),
            "train_min": float(train_weights.min()),
            "train_max": float(train_weights.max()),
            "dev_weight": 1.0,
        },
        "popularity_shortcut_diagnostic": {
            "train_skills_with_grade2": len(grade2_skill_counts),
            "top10_share_of_train_grade2": (
                top10_grade2 / total_train_grade2 if total_train_grade2 else 0.0
            ),
            "top10_train_grade2_skills": [
                {"skill_id": skill_id, "grade2_count": count}
                for skill_id, count in grade2_skill_counts.most_common(10)
            ],
            "required_training_comparison": "NeuMF vs cosine vs train-only skill popularity",
        },
        "split_contract": {
            "gradient_training": "train only",
            "checkpoint_selection": "dev only",
            "official_87_tasks": "evaluation only and unread",
            "family_cross_split": False,
        },
        "candidate_source_feature_names": list(RETRIEVAL_SOURCES),
        "query_source_type_names": list(SOURCE_TYPES),
        "privileged_fields_excluded_from_inputs": [
            "judge_evidence",
            "judge_reason_zh",
            "raw_judge_grade",
            "calibration_action",
            "known_source_pair",
            "source_skill_id",
            "official task",
            "gold skill",
            "verifier",
            "reward",
            "trajectory",
        ],
        "provenance": {
            "pairs_sha256": sha256_file(pairs_path),
            "query_embeddings_sha256": sha256_file(query_embeddings_path),
            "skill_embeddings_sha256": sha256_file(skill_embeddings_path),
            "private_arrays": str(arrays_path.relative_to(ROOT)),
            "private_arrays_sha256": sha256_file(arrays_path),
        },
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    PUBLIC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_REPORT.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
