#!/usr/bin/env python3
"""Train V3 phase NeuMF using train and dev only; never reads test labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from phase_neumf_common import (
    GMF,
    MLP,
    NeuMF,
    NCFConfig,
    PhaseData,
    ROOT,
    TrainConfig,
    checkpoint_payload,
    evaluate_cosine,
    evaluate_model,
    evaluate_popularity,
    preference_pairs,
    seed_all,
    train_one,
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
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gmf-latent-dim", type=int, default=16)
    parser.add_argument(
        "--mlp-layer-sizes",
        type=int,
        nargs="+",
        default=(64, 32, 16),
        help="NeuMF MLP widths, e.g. --mlp-layer-sizes 32 16 8.",
    )
    args = parser.parse_args()
    config = TrainConfig(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        patience=args.patience,
    )
    seed_all(config.seed)
    if (args.output / "internal_test_report.json").exists():
        raise FileExistsError(
            "this output already has a frozen internal-test result; use a new output directory"
        )
    data = PhaseData(args.arrays, config.device)
    # Deliberately reject accidental test access in this process.
    original_indices = data.indices
    def train_dev_indices(split: str):
        if split == "test":
            raise RuntimeError("training process is forbidden from reading internal test")
        return original_indices(split)
    data.indices = train_dev_indices  # type: ignore[method-assign]

    args.output.mkdir(parents=True, exist_ok=True)
    model_config = NCFConfig(
        input_dim=data.input_dim,
        gmf_latent_dim=args.gmf_latent_dim,
        mlp_layer_sizes=tuple(args.mlp_layer_sizes),
    )
    preferences = preference_pairs(data)
    print(f"train pairs={len(data.indices('train'))} preference pairs={len(preferences[0])}", flush=True)

    gmf = GMF(model_config)
    gmf_info = train_one(
        gmf, data, torch.optim.Adam(gmf.parameters(), lr=config.adam_lr), config,
        config.pretrain_epochs, config.seed + 1, preferences,
    )
    mlp = MLP(model_config)
    mlp_info = train_one(
        mlp, data, torch.optim.Adam(mlp.parameters(), lr=config.adam_lr), config,
        config.pretrain_epochs, config.seed + 2, preferences,
    )
    neumf = NeuMF(model_config)
    neumf.load_pretrained(gmf, mlp, alpha=config.alpha)
    neumf_info = train_one(
        neumf, data, torch.optim.SGD(neumf.parameters(), lr=config.sgd_lr), config,
        config.finetune_epochs, config.seed + 3, preferences,
    )
    torch.save(checkpoint_payload(gmf, model_config, config), args.output / "gmf.pt")
    torch.save(checkpoint_payload(mlp, model_config, config), args.output / "mlp.pt")
    torch.save(checkpoint_payload(neumf, model_config, config), args.output / "neumf.pt")
    popularity = train_popularity(data)
    report = {
        "schema_version": "skilldag_ncf.v3.phase_neumf_training.v1",
        "generated_at": utc_now(),
        "api_calls_made": 0,
        "test_split_read": False,
        "initialization": "from_scratch",
        "model_config": asdict(model_config),
        "train_config": asdict(config),
        "counts": {"train_pairs": len(data.indices("train")), "preference_pairs": len(preferences[0])},
        "training": {"gmf": gmf_info, "mlp": mlp_info, "neumf": neumf_info},
        "dev_metrics": {
            "phase_cosine": evaluate_cosine(data, "dev"),
            "skill_popularity": evaluate_popularity(data, "dev", popularity),
            "neumf": evaluate_model(neumf, data, "dev", config.batch_size),
        },
        "checkpoint": str(args.output / "neumf.pt"),
    }
    write_json(args.output / "training_report.json", report)
    print(f"saved {args.output / 'neumf.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
