#!/usr/bin/env python3
"""Train one locked ALFWorld NCF ablation arm on CPU.

The script supports both Skill and Memory arrays through the shared trainer.
It never reads the external retrieval evaluation queries or official rewards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts/ncf_v3", ROOT / "scripts/memp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase_neumf_common import (  # noqa: E402
    GMF,
    MLP,
    NeuMF,
    NCFConfig,
    PhaseData,
    TrainConfig,
    checkpoint_payload,
    evaluate_cosine,
    evaluate_model,
    preference_pairs,
    seed_all,
    train_one,
    utc_now,
    write_json,
)
from train_memory_neumf import MemoryData  # noqa: E402


class LinearScorer(nn.Module):
    """Single linear task-resource interaction over concatenated embeddings."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output = nn.Linear(2 * input_dim, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, query: torch.Tensor, resource: torch.Tensor) -> torch.Tensor:
        return self.output(torch.cat((query, resource), dim=-1)).squeeze(-1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(domain: str, arrays: Path, device: str):
    if domain == "skill":
        data = PhaseData(arrays, device)
        original_indices = data.indices

        def train_dev_only(split: str):
            if split == "test":
                raise RuntimeError("training process is forbidden from reading internal test")
            return original_indices(split)

        data.indices = train_dev_only  # type: ignore[method-assign]
        return data
    if domain == "memory":
        return MemoryData(arrays, device)
    raise ValueError(f"unsupported domain: {domain}")


def binary_preference_pairs(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build positive-vs-negative preferences without imposing Grade 2 > 1."""
    by_query: dict[int, list[int]] = {}
    for row_index in data.indices("train"):
        query_index = int(data.pair_phase_indices[row_index])
        by_query.setdefault(query_index, []).append(int(row_index))
    better: list[int] = []
    worse: list[int] = []
    weights: list[float] = []
    for rows in by_query.values():
        positives = [row for row in rows if int(data.target_grade[row]) > 0]
        negatives = [row for row in rows if int(data.target_grade[row]) == 0]
        for positive in positives:
            for negative in negatives:
                better.append(positive)
                worse.append(negative)
                weights.append(
                    math.sqrt(
                        float(data.sample_weight[positive] * data.sample_weight[negative])
                    )
                )
    weights_array = np.asarray(weights, dtype=np.float32)
    weights_array /= weights_array.mean()
    return (
        np.asarray(better, dtype=np.int64),
        np.asarray(worse, dtype=np.int64),
        weights_array,
    )


def save_linear_checkpoint(
    model: LinearScorer, config: TrainConfig, path: Path
) -> None:
    torch.save(
        {
            "schema_version": "agent_skill_evolution.linear_scorer.v1",
            "state_dict": model.state_dict(),
            "model_config": {"input_dim": model.input_dim},
            "train_config": asdict(config),
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("skill", "memory"), required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "linear",
            "standard",
            "binary",
            "no_pairwise",
            "random_negative",
            "random_negative_matched",
        ),
        required=True,
    )
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device != "cpu":
        raise ValueError("the frozen ablation contract requires CPU training")
    if args.resume and (args.output / "training_report.json").exists():
        print(json.dumps({"output": str(args.output), "status": "already_complete"}))
        return 0
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")

    pairwise_lambda = 0.0 if args.variant == "no_pairwise" else 0.20
    config = TrainConfig(
        seed=args.seed,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        patience=args.patience,
        pairwise_lambda=pairwise_lambda,
        device=args.device,
    )
    seed_all(config.seed)
    data = load_data(args.domain, args.arrays, config.device)
    preferences = (
        binary_preference_pairs(data)
        if args.variant == "binary"
        else preference_pairs(data)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    common: dict[str, Any] = {
        "schema_version": "agent_skill_evolution.alfworld_ncf_ablation_training.v1",
        "generated_at": utc_now(),
        "domain": args.domain,
        "variant": args.variant,
        "arrays": str(args.arrays.resolve()),
        "arrays_sha256": sha256(args.arrays),
        "api_calls_made": 0,
        "external_eval_read": False,
        "device": config.device,
        "train_config": asdict(config),
        "counts": {
            "train_pairs": int(len(data.indices("train"))),
            "dev_pairs": int(len(data.indices("dev"))),
            "preference_pairs_available": int(len(preferences[0])),
            "preference_pairs_optimized": (
                int(len(preferences[0])) if pairwise_lambda > 0.0 else 0
            ),
        },
        "dev_cosine": evaluate_cosine(data, "dev"),
    }

    if args.variant == "linear":
        model = LinearScorer(data.input_dim)
        info = train_one(
            model,
            data,
            torch.optim.Adam(model.parameters(), lr=config.adam_lr),
            config,
            config.pretrain_epochs,
            config.seed + 1,
            preferences,
        )
        save_linear_checkpoint(model, config, args.output / "linear.pt")
        common["model_config"] = {
            "type": "linear_concat",
            "input_dim": data.input_dim,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        common["training"] = {"linear": info}
        common["dev_model"] = evaluate_model(model, data, "dev", config.batch_size)
        common["checkpoint"] = str((args.output / "linear.pt").resolve())
    else:
        model_config = NCFConfig(
            input_dim=data.input_dim,
            gmf_latent_dim=16,
            mlp_layer_sizes=(64, 32, 16),
        )
        def train_or_resume_pretrain(model, name: str, seed: int):
            checkpoint_path = args.output / f"{name}.pt"
            info_path = args.output / f"{name}_training.json"
            if args.resume and checkpoint_path.exists() and info_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["state_dict"])
                return json.loads(info_path.read_text(encoding="utf-8"))
            info = train_one(
                model,
                data,
                torch.optim.Adam(model.parameters(), lr=config.adam_lr),
                config,
                config.pretrain_epochs,
                seed,
                preferences,
            )
            torch.save(checkpoint_payload(model, model_config, config), checkpoint_path)
            write_json(info_path, info)
            return info

        gmf = GMF(model_config)
        gmf_info = train_or_resume_pretrain(gmf, "gmf", config.seed + 1)
        mlp = MLP(model_config)
        mlp_info = train_or_resume_pretrain(mlp, "mlp", config.seed + 2)
        neumf = NeuMF(model_config)
        neumf.load_pretrained(gmf, mlp, alpha=config.alpha)
        neumf_info = train_one(
            neumf,
            data,
            torch.optim.SGD(neumf.parameters(), lr=config.sgd_lr),
            config,
            config.finetune_epochs,
            config.seed + 3,
            preferences,
        )
        torch.save(
            checkpoint_payload(neumf, model_config, config),
            args.output / "neumf.pt",
        )
        common["model_config"] = asdict(model_config)
        common["training"] = {
            "gmf": gmf_info,
            "mlp": mlp_info,
            "neumf": neumf_info,
        }
        common["dev_model"] = evaluate_model(neumf, data, "dev", config.batch_size)
        common["checkpoint"] = str((args.output / "neumf.pt").resolve())

    write_json(args.output / "training_report.json", common)
    print(json.dumps({"output": str(args.output), "dev_model": common["dev_model"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
