#!/usr/bin/env python3
"""Fit train-only skill priors and choose conservative fusion alpha on dev."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from phase_neumf_common import (
    NCFConfig,
    NeuMF,
    PhaseData,
    ROOT,
    cosine_scores,
    model_scores,
    ranking_metrics,
    utc_now,
    write_json,
)


DEFAULT_ARRAYS = ROOT / "data/alfworld_task_skill/task_skill_v3_phase/model_data/content_neumf_v1/arrays.npz"
DEFAULT_MODEL_DIR = ROOT / ".runtime/skilldag_ncf_v3/models/content_neumf_v1"


def normalized_within_phase(data, indices, values):
    result = np.empty(len(indices), dtype=np.float64)
    phases = data.pair_phase_indices[indices]
    for phase in np.unique(phases):
        positions = np.flatnonzero(phases == phase)
        current = values[positions]
        result[positions] = (current - current.mean()) / (current.std() + 1e-8)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    checkpoint = torch.load(args.model_dir / "neumf.pt", map_location="cpu", weights_only=True)
    data = PhaseData(args.arrays)
    model = NeuMF(NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    train = data.indices("train")
    train_logits = model_scores(model, data, train, 512)
    sums = np.zeros(len(data.skill_ids), dtype=np.float64)
    counts = np.zeros(len(data.skill_ids), dtype=np.int64)
    for pair_index, logit in zip(train, train_logits):
        skill_index = int(data.pair_skill_indices[pair_index])
        sums[skill_index] += float(logit)
        counts[skill_index] += 1
    priors = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    dev = data.indices("dev")
    dev_logits = model_scores(model, data, dev, 512)
    specific = dev_logits - priors[data.pair_skill_indices[dev]]
    cosine_z = normalized_within_phase(data, dev, cosine_scores(data, dev))
    ncf_z = normalized_within_phase(data, dev, specific)
    trials = {}
    best = None
    for alpha in (0.25, 0.50, 0.75):
        metrics = ranking_metrics(data, dev, alpha * cosine_z + (1.0 - alpha) * ncf_z)
        trials[str(alpha)] = metrics
        key = (metrics["ndcg@3"], metrics["required_recall@3"], -metrics["bad_item_rate@3"])
        if best is None or key > best[0]:
            best = (key, alpha)
    assert best is not None
    report = {
        "schema_version": "skilldag_ncf.v3.phase_online_calibration.v1",
        "generated_at": utc_now(),
        "api_calls_made": 0,
        "statistics_source": "internal train only",
        "selection_source": "internal dev only",
        "test_split_read": False,
        "alpha_grid": [0.25, 0.50, 0.75],
        "alpha": best[1],
        "dev_trials": trials,
        "skill_logit_prior": {
            str(skill_id): float(prior)
            for skill_id, prior in zip(data.skill_ids, priors)
        },
    }
    write_json(args.model_dir / "online_calibration.json", report)
    print(f"selected alpha={best[1]} on dev; test read=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
