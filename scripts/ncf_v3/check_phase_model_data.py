#!/usr/bin/env python3
"""Audit V3 content-only NeuMF arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data/alfworld_task_skill/task_skill_v3_phase/model_data/content_neumf_v1"


def main() -> int:
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    with np.load(MODEL_DIR / "arrays.npz", allow_pickle=False) as arrays:
        shapes = {name: arrays[name].shape for name in arrays.files}
        if shapes["phase_embeddings"] != (2791, 3072):
            raise ValueError(f"bad phase embedding shape: {shapes['phase_embeddings']}")
        if shapes["skill_embeddings"] != (37, 3072):
            raise ValueError(f"bad skill embedding shape: {shapes['skill_embeddings']}")
        if len(arrays["pair_ids"]) != 49773 or len(set(arrays["pair_ids"].tolist())) != 49773:
            raise ValueError("pair IDs must contain 49,773 unique entries")
        if set(arrays["target_grade"].tolist()) != {0, 1, 2}:
            raise ValueError("target grades must be 0/1/2")
        if not np.allclose(arrays["target_relevance"], arrays["target_grade"] / 2.0):
            raise ValueError("target relevance is not grade/2")
        if not np.isfinite(arrays["phase_embeddings"]).all() or not np.isfinite(arrays["sample_weight"]).all():
            raise ValueError("arrays contain NaN/Inf")
        train_mask = arrays["split_codes"] == 0
        if not np.isclose(arrays["sample_weight"][train_mask].mean(), 1.0, atol=1e-6):
            raise ValueError("train sample weights are not mean-normalized")
        if not np.all(arrays["sample_weight"][~train_mask] == 1.0):
            raise ValueError("dev/test sample weights must stay neutral")
        task_split_sets = {}
        for code, name in ((0, "train"), (1, "dev"), (2, "test")):
            pair_mask = arrays["split_codes"] == code
            task_split_sets[name] = set(arrays["pair_task_indices"][pair_mask].tolist())
        if any(task_split_sets[left] & task_split_sets[right] for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))):
            raise ValueError("task indices overlap across splits")
        if metadata["content_only_contract"]["learned_task_id_embedding"] or metadata["content_only_contract"]["learned_skill_id_embedding"]:
            raise ValueError("model data contract unexpectedly enables learned ID embeddings")
        if metadata["training_weight_contract"]["statistics_source"] != "internal train split only":
            raise ValueError("weights use non-train statistics")
        if metadata["api_calls_made"] != 0:
            raise ValueError("model-data preparation unexpectedly called an API")

    print(f"[OK] Array shapes: {shapes}")
    print(f"[OK] Task splits: {metadata['task_counts_by_split']}")
    print(f"[OK] Pair splits: {metadata['pair_counts_by_split']}")
    print(f"[OK] Grade splits: {metadata['grade_counts_by_split']}")
    print(f"[OK] Train weight range: {metadata['training_weight_contract']['train_min']:.4f} .. {metadata['training_weight_contract']['train_max']:.4f}, mean=1")
    print("[OK] Content-only inputs; IDs are indices; no API or privileged input")
    print("[PASS] V3 NeuMF model arrays are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
