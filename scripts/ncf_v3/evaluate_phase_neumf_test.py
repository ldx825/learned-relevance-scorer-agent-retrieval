#!/usr/bin/env python3
"""One-shot internal-test evaluation of an already frozen V3 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from phase_neumf_common import (
    NCFConfig,
    NeuMF,
    PhaseData,
    ROOT,
    evaluate_cosine,
    evaluate_model,
    evaluate_popularity,
    train_popularity,
    utc_now,
    write_json,
)


DEFAULT_ARRAYS = ROOT / "data/alfworld_task_skill/task_skill_v3_phase/model_data/content_neumf_v1/arrays.npz"
DEFAULT_OUTPUT = ROOT / ".runtime/skilldag_ncf_v3/models/content_neumf_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    test_report = args.output / "internal_test_report.json"
    if test_report.exists():
        raise FileExistsError(
            "internal test was already evaluated for this output; refusing to overwrite it"
        )
    checkpoint = torch.load(args.output / "neumf.pt", map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != "skilldag_ncf.v3.phase_neumf_checkpoint.v1":
        raise ValueError("unexpected checkpoint schema")
    data = PhaseData(args.arrays)
    model = NeuMF(NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    popularity = train_popularity(data)
    metrics = {
        "phase_cosine": evaluate_cosine(data, "test"),
        "skill_popularity": evaluate_popularity(data, "test", popularity),
        "neumf": evaluate_model(model, data, "test", checkpoint["train_config"]["batch_size"]),
    }
    report = {
        "schema_version": "skilldag_ncf.v3.phase_neumf_internal_test.v1",
        "generated_at": utc_now(),
        "api_calls_made": 0,
        "checkpoint_frozen": True,
        "test_evaluation_runs": 1,
        "metrics": metrics,
        "limitations": [
            "Metrics measure phase-level weak-label ranking, not ALFWorld execution success.",
            "Ranking is measured inside the cached candidate pool, not across all 37 skills.",
        ],
    }
    write_json(test_report, report)
    for name, values in metrics.items():
        print(name, f"NDCG@3={values['ndcg@3']:.4f}", f"required@3={values['required_recall@3']:.4f}", f"bad@3={values['bad_item_rate@3']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
