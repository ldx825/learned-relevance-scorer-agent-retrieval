#!/usr/bin/env python3
"""Shared, test-isolated training utilities for phase-aware content NeuMF."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src/GraphOfSkills_NCF"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gos.ncf.models import GMF, MLP, NeuMF, NCFConfig  # noqa: E402


SPLITS = {"train": 0, "dev": 1, "test": 2}


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    batch_size: int = 512
    pair_batch_size: int = 512
    pretrain_epochs: int = 20
    finetune_epochs: int = 20
    patience: int = 5
    adam_lr: float = 1e-3
    sgd_lr: float = 1e-2
    pairwise_lambda: float = 0.20
    alpha: float = 0.5
    device: str = "cpu"


class PhaseData:
    def __init__(self, path: Path, device: str = "cpu") -> None:
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as z:
            for name in z.files:
                setattr(self, name, z[name].copy())
        self.phase_embeddings_t = torch.from_numpy(
            self.phase_embeddings.astype(np.float32)
        ).to(device)
        self.skill_embeddings_t = torch.from_numpy(
            self.skill_embeddings.astype(np.float32)
        ).to(device)
        self.device = device
        if self.phase_embeddings.shape[1] != self.skill_embeddings.shape[1]:
            raise ValueError("phase and skill embedding dimensions differ")

    @property
    def input_dim(self) -> int:
        return int(self.phase_embeddings.shape[1])

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split_codes == SPLITS[split])

    def batch(self, indices: np.ndarray):
        p = torch.as_tensor(self.pair_phase_indices[indices], device=self.device).long()
        s = torch.as_tensor(self.pair_skill_indices[indices], device=self.device).long()
        y = torch.as_tensor(self.target_relevance[indices], device=self.device).float()
        w = torch.as_tensor(self.sample_weight[indices], device=self.device).float()
        return self.phase_embeddings_t[p], self.skill_embeddings_t[s], y, w


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(rank + 2) for rank, g in enumerate(grades))


def ranking_metrics(
    data: PhaseData, indices: np.ndarray, scores: np.ndarray, ks=(3, 5, 10)
) -> dict[str, Any]:
    grouped: dict[int, list[tuple[float, str, int]]] = {}
    for i, score in zip(indices, scores):
        phase = int(data.pair_phase_indices[i])
        skill = int(data.pair_skill_indices[i])
        grouped.setdefault(phase, []).append(
            (float(score), str(data.skill_ids[skill]), int(data.target_grade[i]))
        )
    values: dict[str, list[float]] = {}
    required_eligible = {k: 0 for k in ks}
    for rows in grouped.values():
        ranked = sorted(rows, key=lambda x: (-x[0], x[1]))
        grades = [x[2] for x in ranked]
        ideal = sorted(grades, reverse=True)
        req_total = sum(g == 2 for g in grades)
        first_req = next((r for r, g in enumerate(grades, 1) if g == 2), None)
        if req_total:
            values.setdefault("mrr_required", []).append(1.0 / first_req if first_req else 0.0)
        for k in ks:
            top = grades[:k]
            denom = _dcg(ideal[:k])
            values.setdefault(f"ndcg@{k}", []).append(_dcg(top) / denom if denom else 0.0)
            values.setdefault(f"bad_item_rate@{k}", []).append(
                sum(g == 0 for g in top) / len(top) if top else 0.0
            )
            if req_total:
                required_eligible[k] += 1
                values.setdefault(f"required_recall@{k}", []).append(
                    sum(g == 2 for g in top) / req_total
                )
    result: dict[str, Any] = {"phase_count": len(grouped)}
    result["required_eligible_phase_count"] = required_eligible[ks[0]]
    result.update({k: float(np.mean(v)) if v else None for k, v in sorted(values.items())})
    return result


@torch.no_grad()
def model_scores(model, data: PhaseData, indices: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        phase, skill, _, _ = data.batch(selected)
        chunks.append(model(phase, skill).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def evaluate_model(model, data: PhaseData, split: str, batch_size: int) -> dict[str, Any]:
    indices = data.indices(split)
    return ranking_metrics(data, indices, model_scores(model, data, indices, batch_size))


def cosine_scores(data: PhaseData, indices: np.ndarray) -> np.ndarray:
    phase = data.phase_embeddings[data.pair_phase_indices[indices]]
    skill = data.skill_embeddings[data.pair_skill_indices[indices]]
    numerator = np.sum(phase * skill, axis=1)
    denominator = np.linalg.norm(phase, axis=1) * np.linalg.norm(skill, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def evaluate_cosine(data: PhaseData, split: str) -> dict[str, Any]:
    indices = data.indices(split)
    return ranking_metrics(data, indices, cosine_scores(data, indices))


def train_popularity(data: PhaseData) -> np.ndarray:
    indices = data.indices("train")
    sums = np.zeros(len(data.skill_ids), dtype=np.float64)
    counts = np.zeros(len(data.skill_ids), dtype=np.int64)
    for i in indices:
        skill = int(data.pair_skill_indices[i])
        sums[skill] += float(data.target_grade[i])
        counts[skill] += 1
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def evaluate_popularity(data: PhaseData, split: str, popularity: np.ndarray) -> dict[str, Any]:
    indices = data.indices(split)
    return ranking_metrics(data, indices, popularity[data.pair_skill_indices[indices]])


def preference_pairs(data: PhaseData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build all strict 2>1, 2>0 and 1>0 preferences within train phases."""
    by_phase: dict[int, list[int]] = {}
    for i in data.indices("train"):
        by_phase.setdefault(int(data.pair_phase_indices[i]), []).append(int(i))
    better: list[int] = []
    worse: list[int] = []
    weights: list[float] = []
    for rows in by_phase.values():
        for left in rows:
            for right in rows:
                gap = int(data.target_grade[left]) - int(data.target_grade[right])
                if gap <= 0:
                    continue
                better.append(left)
                worse.append(right)
                weights.append(
                    math.sqrt(float(data.sample_weight[left] * data.sample_weight[right]))
                    * gap / 2.0
                )
    weights_np = np.asarray(weights, dtype=np.float32)
    weights_np /= weights_np.mean()
    return np.asarray(better), np.asarray(worse), weights_np


def train_one(
    model,
    data: PhaseData,
    optimizer,
    config: TrainConfig,
    epochs: int,
    seed: int,
    preferences: tuple[np.ndarray, np.ndarray, np.ndarray],
    selection_metric: str = "ndcg",
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_indices = data.indices("train")
    better, worse, pair_weights = preferences
    best_state = None
    best_key = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        point_order = rng.permutation(train_indices)
        point_sum = 0.0
        for start in range(0, len(point_order), config.batch_size):
            selected = point_order[start : start + config.batch_size]
            phase, skill, target, weight = data.batch(selected)
            optimizer.zero_grad(set_to_none=True)
            raw = F.binary_cross_entropy_with_logits(model(phase, skill), target, reduction="none")
            loss = (raw * weight).sum() / weight.sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()
            point_sum += float(loss.detach()) * len(selected)

        pair_sum = 0.0
        if config.pairwise_lambda > 0.0:
            pair_order = rng.permutation(len(better))
            for start in range(0, len(pair_order), config.pair_batch_size):
                selected = pair_order[start : start + config.pair_batch_size]
                hi = better[selected]
                lo = worse[selected]
                phase_hi, skill_hi, _, _ = data.batch(hi)
                phase_lo, skill_lo, _, _ = data.batch(lo)
                weight = torch.as_tensor(pair_weights[selected], device=data.device)
                optimizer.zero_grad(set_to_none=True)
                raw = F.softplus(-(model(phase_hi, skill_hi) - model(phase_lo, skill_lo)))
                loss = config.pairwise_lambda * (raw * weight).sum() / weight.sum().clamp_min(1e-12)
                loss.backward()
                optimizer.step()
                pair_sum += float(loss.detach()) * len(selected)

        if selection_metric == "final":
            history.append({
                "epoch": epoch,
                "pointwise_loss": point_sum / len(train_indices),
                "weighted_pairwise_loss": pair_sum / len(better) if len(better) else 0.0,
            })
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            print(
                f"[{model.__class__.__name__}] epoch={epoch} final-fit "
                f"point={point_sum / len(train_indices):.4f} "
                f"pair={pair_sum / len(better) if len(better) else 0.0:.4f}",
                flush=True,
            )
            continue
        dev = evaluate_model(model, data, "dev", config.batch_size)
        history.append({
            "epoch": epoch,
            "pointwise_loss": point_sum / len(train_indices),
            "weighted_pairwise_loss": pair_sum / len(better) if len(better) else 0.0,
            "dev": dev,
        })
        # Select only on internal dev. The MRR option targets first-hit rank;
        # the historical NDCG behavior remains the default.
        if selection_metric == "mrr":
            key = (dev["mrr_required"], dev["ndcg@3"], dev["required_recall@3"])
        elif selection_metric == "ndcg":
            key = (dev["ndcg@3"], dev["required_recall@3"], -dev["bad_item_rate@3"])
        elif selection_metric != "final":
            raise ValueError(f"unsupported selection metric: {selection_metric}")
        if selection_metric == "mrr":
            message = (
                f"mrr={dev['mrr_required']:.4f} ndcg@3={dev['ndcg@3']:.4f} "
                f"required@3={dev['required_recall@3']:.4f}"
            )
        else:
            message = (
                f"ndcg@3={dev['ndcg@3']:.4f} required@3={dev['required_recall@3']:.4f} "
                f"bad@3={dev['bad_item_rate@3']:.4f}"
            )
        print(f"[{model.__class__.__name__}] epoch={epoch} {message}", flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_dev_selection_key": best_key, "history": history}


def checkpoint_payload(model, model_config: NCFConfig, train_config: TrainConfig) -> dict[str, Any]:
    return {
        "schema_version": "skilldag_ncf.v3.phase_neumf_checkpoint.v1",
        "state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
