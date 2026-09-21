#!/usr/bin/env python3
"""Create content-only NeuMF arrays from leakage-safe V3 phase pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/alfworld_task_skill"
V3 = DATA_ROOT / "task_skill_v3_phase"
PHASES = V3 / "phases/phases.jsonl"
PAIRS = V3 / "labeled_pairs/pairs.jsonl"
PHASE_CACHE = V3 / "candidates/embedding_cache/text-embedding-3-large"
SKILL_CACHE = DATA_ROOT / "shared/embeddings/skill_embeddings_37.json"
OUTPUT = V3 / "model_data/content_neumf_v1"
SPLIT_CODES = {"train": 0, "dev": 1, "test": 2}
PHASE_GROUPS = ("locate", "transform", "place", "verify")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_phase_cache() -> dict[str, dict[str, Any]]:
    cache = {}
    for path in sorted(PHASE_CACHE.glob("shard_*.json")):
        for phase_id, entry in json.loads(path.read_text(encoding="utf-8")).items():
            if phase_id in cache and cache[phase_id] != entry:
                raise ValueError(f"conflicting phase embedding: {phase_id}")
            cache[phase_id] = entry
    return cache


def main() -> int:
    phases = load_jsonl(PHASES)
    pairs = load_jsonl(PAIRS)
    phase_by_id = {row["phase_id"]: row for row in phases}
    phase_cache = load_phase_cache()
    skill_cache = json.loads(SKILL_CACHE.read_text(encoding="utf-8"))
    if set(phase_by_id) != set(phase_cache):
        raise ValueError("phase embedding cache does not exactly match phase records")
    models = {entry["model"] for entry in phase_cache.values()} | {entry["model"] for entry in skill_cache.values()}
    if models != {"text-embedding-3-large"}:
        raise ValueError(f"unexpected embedding models: {models}")

    phase_ids = sorted(phase_by_id)
    skill_ids = sorted(skill_cache)
    task_ids = sorted({row["task_record_id"] for row in phases})
    phase_index = {value: index for index, value in enumerate(phase_ids)}
    skill_index = {value: index for index, value in enumerate(skill_ids)}
    task_index = {value: index for index, value in enumerate(task_ids)}
    phase_embeddings = np.asarray([phase_cache[value]["embedding"] for value in phase_ids], dtype=np.float32)
    skill_embeddings = np.asarray([skill_cache[value]["embedding"] for value in skill_ids], dtype=np.float32)
    if phase_embeddings.shape != (2791, 3072) or skill_embeddings.shape != (37, 3072):
        raise ValueError(f"unexpected embedding shapes: {phase_embeddings.shape}, {skill_embeddings.shape}")

    train_pairs = [row for row in pairs if row["internal_split"] == "train"]
    train_task_frequency = Counter(row["task_record_id"] for row in train_pairs)
    train_group_frequency = Counter(row["phase_group"] for row in train_pairs)
    train_skill_frequency = Counter(row["skill_id"] for row in train_pairs)
    train_pair_count = len(train_pairs)
    group_factor = {
        group: train_pair_count / (len(PHASE_GROUPS) * train_group_frequency[group])
        for group in PHASE_GROUPS
    }
    skill_factor = {
        skill_id: np.sqrt(train_pair_count / train_skill_frequency[skill_id])
        for skill_id in skill_ids
    }

    pair_phase_indices = np.empty(len(pairs), dtype=np.int32)
    pair_skill_indices = np.empty(len(pairs), dtype=np.int16)
    pair_task_indices = np.empty(len(pairs), dtype=np.int16)
    target_grade = np.empty(len(pairs), dtype=np.int8)
    target_relevance = np.empty(len(pairs), dtype=np.float32)
    label_confidence = np.empty(len(pairs), dtype=np.float32)
    sample_weight = np.ones(len(pairs), dtype=np.float32)
    split_codes = np.empty(len(pairs), dtype=np.int8)
    phase_group_codes = np.empty(len(pairs), dtype=np.int8)
    source_features = np.empty((len(pairs), 3), dtype=np.float32)
    pair_ids = []
    group_code = {value: index for index, value in enumerate(PHASE_GROUPS)}

    raw_train_weights = []
    train_offsets = []
    for offset, row in enumerate(pairs):
        pair_phase_indices[offset] = phase_index[row["phase_id"]]
        pair_skill_indices[offset] = skill_index[row["skill_id"]]
        pair_task_indices[offset] = task_index[row["task_record_id"]]
        grade = int(row["target_grade"])
        target_grade[offset] = grade
        target_relevance[offset] = grade / 2.0
        label_confidence[offset] = float(row["label_confidence"])
        split_codes[offset] = SPLIT_CODES[row["internal_split"]]
        phase_group_codes[offset] = group_code[row["phase_group"]]
        sources = set(row["candidate_sources"])
        source_features[offset] = [
            float("phase_cosine" in sources),
            float("full_task_cosine" in sources),
            float("v2_ncf_hard" in sources),
        ]
        pair_ids.append(row["pair_id"])
        if row["internal_split"] == "train":
            weight = (
                float(row["label_confidence"])
                / train_task_frequency[row["task_record_id"]]
                * group_factor[row["phase_group"]]
                * skill_factor[row["skill_id"]]
            )
            raw_train_weights.append(weight)
            train_offsets.append(offset)
    normalizer = float(np.mean(raw_train_weights))
    for offset, weight in zip(train_offsets, raw_train_weights, strict=True):
        sample_weight[offset] = weight / normalizer

    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays_path = OUTPUT / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        task_ids=np.asarray(task_ids),
        phase_ids=np.asarray(phase_ids),
        skill_ids=np.asarray(skill_ids),
        pair_ids=np.asarray(pair_ids),
        phase_embeddings=phase_embeddings,
        skill_embeddings=skill_embeddings,
        pair_task_indices=pair_task_indices,
        pair_phase_indices=pair_phase_indices,
        pair_skill_indices=pair_skill_indices,
        target_grade=target_grade,
        target_relevance=target_relevance,
        label_confidence=label_confidence,
        sample_weight=sample_weight,
        split_codes=split_codes,
        phase_group_codes=phase_group_codes,
        candidate_source_features=source_features,
    )
    split_pair_counts = Counter(row["internal_split"] for row in pairs)
    split_phase_counts = Counter(row["internal_split"] for row in phases)
    split_task_counts = {
        split: len({row["task_record_id"] for row in pairs if row["internal_split"] == split})
        for split in SPLIT_CODES
    }
    grade_by_split = defaultdict(Counter)
    for row in pairs:
        grade_by_split[row["internal_split"]][str(row["target_grade"])] += 1
    train_weights = sample_weight[split_codes == SPLIT_CODES["train"]]
    metadata = {
        "schema_version": "skilldag_ncf.v3.content_neumf_model_data.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_made": 0,
        "embedding_model": "text-embedding-3-large",
        "embedding_dimension": 3072,
        "task_count": len(task_ids),
        "phase_count": len(phase_ids),
        "skill_count": len(skill_ids),
        "pair_count": len(pairs),
        "task_counts_by_split": split_task_counts,
        "phase_counts_by_split": dict(sorted(split_phase_counts.items())),
        "pair_counts_by_split": dict(sorted(split_pair_counts.items())),
        "grade_counts_by_split": {split: dict(sorted(counts.items())) for split, counts in sorted(grade_by_split.items())},
        "split_codes": SPLIT_CODES,
        "phase_group_codes": group_code,
        "target_relevance": "target_grade / 2",
        "content_only_contract": {
            "learned_task_id_embedding": False,
            "learned_skill_id_embedding": False,
            "model_inputs": ["phase_context_embedding", "skill_document_embedding"],
            "ids_used_only_as_array_indices": True,
        },
        "training_weight_contract": {
            "statistics_source": "internal train split only",
            "formula": "confidence / task_pair_count * inverse_phase_group_frequency * inverse_sqrt_skill_frequency",
            "normalized_train_mean": float(train_weights.mean()),
            "train_min": float(train_weights.min()),
            "train_max": float(train_weights.max()),
            "dev_test_weight": 1.0,
        },
        "candidate_source_feature_names": ["phase_cosine", "full_task_cosine", "v2_ncf_hard"],
        "privileged_fields_excluded_from_inputs": ["judge_reason", "raw_judge_grade", "calibration_action", "expert_plan", "reward", "trajectory"],
        "provenance": {
            "phases_sha256": sha256_file(PHASES),
            "pairs_sha256": sha256_file(PAIRS),
            "skill_embeddings_sha256": sha256_file(SKILL_CACHE),
            "phase_embedding_cache": str(PHASE_CACHE),
        },
        "outputs": {"arrays": "arrays.npz", "metadata": "metadata.json"},
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
