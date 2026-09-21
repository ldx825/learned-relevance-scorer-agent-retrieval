"""Deterministic training and evaluation for the NeuMF-graded experiment."""

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
from torch import Tensor, nn

from .losses import PointwiseNCFLoss
from .models import GMF, MLP, NeuMF, NCFConfig


SPLIT_CODES = {"train": 0, "dev": 1, "test": 2}


@dataclass(frozen=True)
class NeuMFGradedTrainingConfig:
    """Paper-aligned optimization settings for our graded supervision."""

    seed: int = 2026
    batch_size: int = 256
    pretrain_epochs: int = 30
    finetune_epochs: int = 30
    patience: int = 6
    adam_learning_rate: float = 1e-3
    sgd_learning_rate: float = 1e-2
    weight_decay: float = 0.0
    alpha: float = 0.5
    device: str = "cpu"
    target_mode: str = "graded"

    def __post_init__(self) -> None:
        if self.target_mode != "graded":
            raise ValueError("this preset intentionally requires target_mode='graded'")
        if self.batch_size < 1 or self.pretrain_epochs < 1 or self.finetune_epochs < 1:
            raise ValueError("batch size and epoch counts must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.adam_learning_rate <= 0 or self.sgd_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")


class NCFArrayData:
    """Validated tensor view over ``task_skill_v1/arrays.npz``."""

    def __init__(self, arrays_path: Path | str, *, device: str = "cpu") -> None:
        self.arrays_path = Path(arrays_path).expanduser().resolve()
        with np.load(self.arrays_path, allow_pickle=False) as arrays:
            self.task_ids = arrays["task_ids"].copy()
            self.skill_ids = arrays["skill_ids"].copy()
            self.pair_ids = arrays["pair_ids"].copy()
            self.task_embeddings = torch.from_numpy(
                arrays["task_embeddings"].astype(np.float32, copy=True)
            ).to(device)
            self.skill_embeddings = torch.from_numpy(
                arrays["skill_embeddings"].astype(np.float32, copy=True)
            ).to(device)
            self.pair_task_indices = arrays["pair_task_indices"].astype(
                np.int64, copy=True
            )
            self.pair_skill_indices = arrays["pair_skill_indices"].astype(
                np.int64, copy=True
            )
            self.pair_features = arrays["pair_features"].astype(np.float32, copy=True)
            self.target_grade = torch.from_numpy(
                arrays["target_grade"].astype(np.int64, copy=True)
            ).to(device)
            self.target_relevance = torch.from_numpy(
                arrays["target_relevance"].astype(np.float32, copy=True)
            ).to(device)
            self.sample_weight = torch.from_numpy(
                arrays["sample_weight"].astype(np.float32, copy=True)
            ).to(device)
            self.split_codes = arrays["split_codes"].astype(np.int8, copy=True)

        pair_count = len(self.pair_ids)
        pair_lengths = {
            len(self.pair_task_indices),
            len(self.pair_skill_indices),
            len(self.target_grade),
            len(self.target_relevance),
            len(self.sample_weight),
            len(self.split_codes),
            len(self.pair_features),
        }
        if pair_lengths != {pair_count}:
            raise ValueError("pair arrays have inconsistent lengths")
        if self.task_embeddings.ndim != 2 or self.skill_embeddings.ndim != 2:
            raise ValueError("task and skill embeddings must be matrices")
        if self.task_embeddings.shape[1] != self.skill_embeddings.shape[1]:
            raise ValueError("task and skill embedding dimensions differ")

    @property
    def input_dim(self) -> int:
        return int(self.task_embeddings.shape[1])

    def indices(self, split: str, *, trainable_only: bool = True) -> np.ndarray:
        if split not in SPLIT_CODES:
            raise ValueError(f"unknown split: {split}")
        mask = self.split_codes == SPLIT_CODES[split]
        if trainable_only:
            mask &= self.target_grade.detach().cpu().numpy() >= 0
        return np.flatnonzero(mask)

    def batch(self, pair_indices: np.ndarray) -> tuple[Tensor, ...]:
        task_indices = torch.as_tensor(
            self.pair_task_indices[pair_indices],
            dtype=torch.long,
            device=self.task_embeddings.device,
        )
        skill_indices = torch.as_tensor(
            self.pair_skill_indices[pair_indices],
            dtype=torch.long,
            device=self.skill_embeddings.device,
        )
        selected = torch.as_tensor(
            pair_indices, dtype=torch.long, device=self.target_grade.device
        )
        return (
            self.task_embeddings[task_indices],
            self.skill_embeddings[skill_indices],
            self.target_grade[selected],
            self.target_relevance[selected],
            self.sample_weight[selected],
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def _ranking_metrics(
    data: NCFArrayData,
    pair_indices: np.ndarray,
    scores: np.ndarray,
    *,
    ks: tuple[int, ...] = (3, 5, 10),
) -> dict[str, float]:
    by_task: dict[int, list[tuple[float, str, int]]] = {}
    grades = data.target_grade.detach().cpu().numpy()
    for pair_index, score in zip(pair_indices, scores):
        grade = int(grades[pair_index])
        if grade < 0:
            continue
        task_index = int(data.pair_task_indices[pair_index])
        skill_index = int(data.pair_skill_indices[pair_index])
        by_task.setdefault(task_index, []).append(
            (float(score), str(data.skill_ids[skill_index]), grade)
        )

    accumulated: dict[str, list[float]] = {}
    for rows in by_task.values():
        ranked = sorted(rows, key=lambda row: (-row[0], row[1]))
        ranked_grades = [row[2] for row in ranked]
        ideal = sorted(ranked_grades, reverse=True)
        required_total = sum(grade == 2 for grade in ranked_grades)
        first_required = next(
            (rank for rank, grade in enumerate(ranked_grades, 1) if grade == 2), None
        )
        accumulated.setdefault("mrr_required", []).append(
            1.0 / first_required if first_required else 0.0
        )
        for k in ks:
            top = ranked_grades[:k]
            ideal_dcg = _dcg(ideal[:k])
            values = {
                f"ndcg@{k}": _dcg(top) / ideal_dcg if ideal_dcg else 0.0,
                f"required_recall@{k}": (
                    sum(grade == 2 for grade in top) / required_total
                    if required_total
                    else 0.0
                ),
                f"bad_item_rate@{k}": (
                    sum(grade == 0 for grade in top) / len(top) if top else 0.0
                ),
            }
            for name, value in values.items():
                accumulated.setdefault(name, []).append(value)
    return {
        "task_count": float(len(by_task)),
        **{
            name: float(sum(values) / len(values)) if values else 0.0
            for name, values in sorted(accumulated.items())
        },
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: NCFArrayData,
    split: str,
    *,
    batch_size: int = 512,
) -> dict[str, float]:
    model.eval()
    pair_indices = data.indices(split, trainable_only=True)
    scores: list[np.ndarray] = []
    for start in range(0, len(pair_indices), batch_size):
        selected = pair_indices[start : start + batch_size]
        task_embeddings, skill_embeddings, _, _, _ = data.batch(selected)
        scores.append(model(task_embeddings, skill_embeddings).cpu().numpy())
    merged = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    return _ranking_metrics(data, pair_indices, merged)


def evaluate_cosine(data: NCFArrayData, split: str) -> dict[str, float]:
    pair_indices = data.indices(split, trainable_only=True)
    return _ranking_metrics(data, pair_indices, data.pair_features[pair_indices, 0])


def evaluate_skill_popularity(data: NCFArrayData, split: str) -> dict[str, Any]:
    """Task-agnostic baseline fitted only on train grades.

    A learned interaction model is not useful if a global per-skill mean grade
    ranks candidates just as well.  Keeping this diagnostic next to NeuMF is
    essential for detecting label or candidate-pool popularity shortcuts.
    """
    train_indices = data.indices("train", trainable_only=True)
    grades = data.target_grade.detach().cpu().numpy()
    grade_sums = np.zeros(len(data.skill_ids), dtype=np.float64)
    counts = np.zeros(len(data.skill_ids), dtype=np.int64)
    for pair_index in train_indices:
        skill_index = int(data.pair_skill_indices[pair_index])
        grade_sums[skill_index] += float(grades[pair_index])
        counts[skill_index] += 1
    means = np.divide(
        grade_sums,
        counts,
        out=np.zeros_like(grade_sums),
        where=counts > 0,
    )
    pair_indices = data.indices(split, trainable_only=True)
    metrics = _ranking_metrics(
        data, pair_indices, means[data.pair_skill_indices[pair_indices]]
    )
    order = np.lexsort((data.skill_ids.astype(str), -means))
    return {
        **metrics,
        "top_train_skills": [
            {
                "skill_id": str(data.skill_ids[index]),
                "mean_grade": float(means[index]),
                "train_pair_count": int(counts[index]),
            }
            for index in order[:10]
        ],
    }


def _train_one_model(
    model: nn.Module,
    data: NCFArrayData,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
) -> dict[str, Any]:
    criterion = PointwiseNCFLoss("graded")
    train_indices = data.indices("train", trainable_only=True)
    rng = np.random.default_rng(seed)
    best_state: dict[str, Tensor] | None = None
    best_ndcg = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        weighted_loss = 0.0
        example_count = 0
        for start in range(0, len(shuffled), batch_size):
            selected = shuffled[start : start + batch_size]
            task_embeddings, skill_embeddings, grades, relevance, weights = data.batch(
                selected
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(task_embeddings, skill_embeddings)
            loss = criterion(
                logits,
                grades,
                target_relevance=relevance,
                sample_weight=weights,
            )
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(selected)
            example_count += len(selected)

        dev = evaluate_model(model, data, "dev", batch_size=batch_size)
        train_loss = weighted_loss / max(example_count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev": dev})
        ndcg = dev["ndcg@3"]
        if ndcg > best_ndcg + 1e-12:
            best_ndcg = ndcg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_dev_ndcg@3": best_ndcg,
        "epochs_ran": len(history),
        "history": history,
    }


def train_neumf_graded(
    arrays_path: Path | str,
    output_dir: Path | str,
    *,
    model_config: NCFConfig | None = None,
    training_config: NeuMFGradedTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train GMF, MLP, and pretrained NeuMF using 2/1/0 graded labels."""
    training_config = training_config or NeuMFGradedTrainingConfig()
    _seed_everything(training_config.seed)
    data = NCFArrayData(arrays_path, device=training_config.device)
    model_config = model_config or NCFConfig(input_dim=data.input_dim)
    if model_config.input_dim != data.input_dim:
        raise ValueError("model input dimension does not match arrays.npz")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gmf = GMF(model_config).to(training_config.device)
    gmf_optimizer = torch.optim.Adam(
        gmf.parameters(),
        lr=training_config.adam_learning_rate,
        weight_decay=training_config.weight_decay,
    )
    gmf_training = _train_one_model(
        gmf,
        data,
        gmf_optimizer,
        epochs=training_config.pretrain_epochs,
        batch_size=training_config.batch_size,
        patience=training_config.patience,
        seed=training_config.seed + 1,
    )

    mlp = MLP(model_config).to(training_config.device)
    mlp_optimizer = torch.optim.Adam(
        mlp.parameters(),
        lr=training_config.adam_learning_rate,
        weight_decay=training_config.weight_decay,
    )
    mlp_training = _train_one_model(
        mlp,
        data,
        mlp_optimizer,
        epochs=training_config.pretrain_epochs,
        batch_size=training_config.batch_size,
        patience=training_config.patience,
        seed=training_config.seed + 2,
    )

    neumf = NeuMF(model_config).to(training_config.device)
    neumf.load_pretrained(gmf, mlp, alpha=training_config.alpha)
    # The paper fine-tunes pretrained NeuMF with vanilla SGD, not Adam.
    neumf_optimizer = torch.optim.SGD(
        neumf.parameters(),
        lr=training_config.sgd_learning_rate,
        weight_decay=training_config.weight_decay,
    )
    neumf_training = _train_one_model(
        neumf,
        data,
        neumf_optimizer,
        epochs=training_config.finetune_epochs,
        batch_size=training_config.batch_size,
        patience=training_config.patience,
        seed=training_config.seed + 3,
    )

    torch.save(gmf.state_dict(), output_dir / "gmf_graded.pt")
    torch.save(mlp.state_dict(), output_dir / "mlp_graded.pt")
    torch.save(neumf.state_dict(), output_dir / "neumf_graded.pt")
    report = {
        "schema_version": "gos_ncf.neumf_graded.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_made": 0,
        "model": "NeuMF",
        "target_mode": "graded",
        "grade_mapping": {"required_2": 1.0, "helpful_1": 0.5, "negative_0": 0.0},
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "parameter_counts": {
            "gmf": sum(parameter.numel() for parameter in gmf.parameters()),
            "mlp": sum(parameter.numel() for parameter in mlp.parameters()),
            "neumf": sum(parameter.numel() for parameter in neumf.parameters()),
        },
        "training": {
            "gmf": gmf_training,
            "mlp": mlp_training,
            "neumf": neumf_training,
        },
        "metrics": {
            "cosine_dev": evaluate_cosine(data, "dev"),
            "cosine_test": evaluate_cosine(data, "test"),
            "skill_popularity_dev": evaluate_skill_popularity(data, "dev"),
            "skill_popularity_test": evaluate_skill_popularity(data, "test"),
            "neumf_dev": evaluate_model(
                neumf, data, "dev", batch_size=training_config.batch_size
            ),
            "neumf_test": evaluate_model(
                neumf, data, "test", batch_size=training_config.batch_size
            ),
        },
        "limitations": [
            "Candidate pairs come from the Judge-labeled retrieval candidate pool; unlabeled skills are not evaluated as implicit negatives.",
            "Metrics measure agreement with weak labels, not ALFWorld execution success.",
            "GoS PPR and runtime reranking are not integrated in this stage.",
            "The task-agnostic skill-popularity baseline must be beaten before claiming task-skill interaction learning.",
        ],
        "outputs": {
            "gmf": str(output_dir / "gmf_graded.pt"),
            "mlp": str(output_dir / "mlp_graded.pt"),
            "neumf": str(output_dir / "neumf_graded.pt"),
            "report": str(output_dir / "report.json"),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
