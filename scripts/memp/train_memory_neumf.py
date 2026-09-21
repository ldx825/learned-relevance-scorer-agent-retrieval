#!/usr/bin/env python3
"""Train content-based GMF, MLP and NeuMF on Memory-Task labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
NCF_UTILS = ROOT / "scripts/ncf_v3"
if str(NCF_UTILS) not in sys.path:
    sys.path.insert(0, str(NCF_UTILS))

from phase_neumf_common import (  # noqa: E402
    GMF,
    MLP,
    NeuMF,
    NCFConfig,
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


class MemoryData:
    """Adapter exposing Memory arrays to the shared Task-Resource trainer."""

    SPLITS = {"train": 0, "dev": 1}

    def __init__(self, path: Path, device: str = "cpu") -> None:
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as payload:
            for name in payload.files:
                setattr(self, name, payload[name].copy())
        self.phase_embeddings = self.task_embeddings
        self.skill_embeddings = self.memory_embeddings
        self.pair_phase_indices = self.pair_task_indices
        self.pair_skill_indices = self.pair_memory_indices
        self.skill_ids = self.memory_ids
        self.phase_embeddings_t = torch.from_numpy(self.task_embeddings.astype(np.float32)).to(device)
        self.skill_embeddings_t = torch.from_numpy(self.memory_embeddings.astype(np.float32)).to(device)
        self.device = device
        if self.task_embeddings.shape[1] != self.memory_embeddings.shape[1]:
            raise ValueError("task and memory embedding dimensions differ")

    @property
    def input_dim(self) -> int:
        return int(self.task_embeddings.shape[1])

    def indices(self, split: str) -> np.ndarray:
        if split == "test":
            raise RuntimeError("training process is forbidden from reading official Dev/Test")
        if split not in self.SPLITS:
            raise ValueError(f"unknown internal split: {split}")
        return np.flatnonzero(self.split_codes == self.SPLITS[split])

    def batch(self, indices: np.ndarray):
        task = torch.as_tensor(self.pair_task_indices[indices], device=self.device).long()
        memory = torch.as_tensor(self.pair_memory_indices[indices], device=self.device).long()
        target = torch.as_tensor(self.target_relevance[indices], device=self.device).float()
        weight = torch.as_tensor(self.sample_weight[indices], device=self.device).float()
        return self.phase_embeddings_t[task], self.skill_embeddings_t[memory], target, weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed GMF/MLP stage checkpoints from the output directory.",
    )
    parser.add_argument(
        "--preference-manifest",
        type=Path,
        help="Optional balanced train-only anchor/rescue ranking preferences.",
    )
    parser.add_argument(
        "--preference-mode", choices=("replace", "append"), default="replace"
    )
    args = parser.parse_args()
    wall_start = time.perf_counter()
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    config = TrainConfig(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        patience=args.patience,
        device=args.device,
    )
    seed_all(config.seed)
    if (args.output / "training_report.json").exists():
        raise FileExistsError("training output already exists; use a new output directory")
    data = MemoryData(args.arrays, config.device)
    args.output.mkdir(parents=True, exist_ok=True)
    model_config = NCFConfig(
        input_dim=data.input_dim,
        gmf_latent_dim=16,
        mlp_layer_sizes=(64, 32, 16),
    )
    generic_preferences = preference_pairs(data)
    preferences = generic_preferences
    manifest_counts: dict[str, int] = {}
    if args.preference_manifest:
        rows = [
            json.loads(line)
            for line in args.preference_manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
        task_positions: dict[int, list[int]] = {}
        source_task_ids = (
            data.parent_task_ids if hasattr(data, "parent_task_ids") else data.task_ids
        )
        for index, task_id in enumerate(source_task_ids):
            task_positions.setdefault(int(task_id), []).append(index)
        row_position = {
            (int(task_index), int(memory_index)): offset
            for offset, (task_index, memory_index) in enumerate(
                zip(data.pair_task_indices, data.pair_memory_indices, strict=True)
            )
        }
        better: list[int] = []
        worse: list[int] = []
        for row in rows:
            if row.get("source_split") != "internal_train" or row.get("validation_or_test_used") is not False:
                raise ValueError("preference manifest contains non-train or evaluation-derived row")
            positions = task_positions[int(row["task_id"])]
            matched = 0
            for task_index in positions:
                positive = row_position.get((task_index, int(row["positive_memory_id"])))
                negative = row_position.get((task_index, int(row["negative_memory_id"])))
                if positive is None or negative is None:
                    continue
                if data.split_codes[positive] != 0 or data.split_codes[negative] != 0:
                    raise ValueError("preference endpoint is not internal_train")
                if data.target_grade[positive] <= data.target_grade[negative]:
                    raise ValueError("manifest preference contradicts array labels")
                better.append(positive)
                worse.append(negative)
                matched += 1
            if not matched:
                raise ValueError(
                    f"preference endpoints missing from arrays: task={row['task_id']} "
                    f"positive={row['positive_memory_id']} negative={row['negative_memory_id']}"
                )
            kind = str(row["pair_type"])
            manifest_counts[kind] = manifest_counts.get(kind, 0) + matched
        manifest_preferences = (
            np.asarray(better, dtype=np.int64),
            np.asarray(worse, dtype=np.int64),
            np.ones(len(better), dtype=np.float32),
        )
        if args.preference_mode == "replace":
            preferences = manifest_preferences
        else:
            preferences = tuple(
                np.concatenate((generic_preferences[index], manifest_preferences[index]))
                for index in range(3)
            )
    print(
        f"train pairs={len(data.indices('train'))} "
        f"dev pairs={len(data.indices('dev'))} preference pairs={len(preferences[0])}",
        flush=True,
    )

    def train_or_resume_pretrain(model, name: str, seed: int):
        checkpoint_path = args.output / f"{name}.pt"
        info_path = args.output / f"{name}_training.json"
        if args.resume and checkpoint_path.exists() and info_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            info = json.loads(info_path.read_text(encoding="utf-8"))
            print(f"[{name.upper()}] resumed {checkpoint_path}", flush=True)
            return info
        info = train_one(
            model,
            data,
            torch.optim.Adam(model.parameters(), lr=config.adam_lr),
            config,
            config.pretrain_epochs,
            seed,
            preferences,
        )
        # Persist each independent pretraining stage immediately.  A CPU job can
        # then resume without repeating a completed stage after interruption.
        torch.save(checkpoint_payload(model, model_config, config), checkpoint_path)
        write_json(info_path, info)
        print(f"[{name.upper()}] saved {checkpoint_path}", flush=True)
        return info

    gmf = GMF(model_config).to(config.device)
    gmf_info = train_or_resume_pretrain(gmf, "gmf", config.seed + 1)
    mlp = MLP(model_config).to(config.device)
    mlp_info = train_or_resume_pretrain(mlp, "mlp", config.seed + 2)
    neumf = NeuMF(model_config).to(config.device)
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

    torch.save(checkpoint_payload(neumf, model_config, config), args.output / "neumf.pt")
    popularity = train_popularity(data)
    report = {
        "schema_version": "task_resource_ncf.memory_neumf_training.v1",
        "generated_at": utc_now(),
        "api_calls_made": 0,
        "official_dev_or_test_read": False,
        "initialization": "from_scratch",
        "model_config": asdict(model_config),
        "train_config": asdict(config),
        "counts": {
            "train_pairs": len(data.indices("train")),
            "model_dev_pairs": len(data.indices("dev")),
            "preference_pairs": len(preferences[0]),
            "generic_preference_pairs": len(generic_preferences[0]),
            "manifest_preference_pairs": sum(manifest_counts.values()),
            "manifest_preference_counts_by_type": manifest_counts,
        },
        "pairwise_supervision": {
            "mode": args.preference_mode if args.preference_manifest else "generic_all_strict_grades",
            "manifest": str(args.preference_manifest.resolve()) if args.preference_manifest else None,
            "sample_weight": "uniform for manifest preferences",
        },
        "training": {"gmf": gmf_info, "mlp": mlp_info, "neumf": neumf_info},
        "model_dev_metrics": {
            "content_cosine": evaluate_cosine(data, "dev"),
            "memory_popularity": evaluate_popularity(data, "dev", popularity),
            "neumf": evaluate_model(neumf, data, "dev", config.batch_size),
        },
        "checkpoint": str(args.output / "neumf.pt"),
    }
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
    report["efficiency"] = {
        "wall_time_seconds": time.perf_counter() - wall_start,
        "device": args.device,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024.0 * 1024.0)
            if str(args.device).startswith("cuda") else 0.0
        ),
    }
    write_json(args.output / "training_report.json", report)
    print(f"saved {args.output / 'neumf.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
