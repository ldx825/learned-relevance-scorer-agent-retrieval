"""Pointwise log-loss objectives for task-skill NCF."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TargetMode = Literal["binary", "graded", "relevance"]


class PointwiseNCFLoss(nn.Module):
    """Weighted binary cross-entropy used by the NCF paper.

    Target modes:

    * ``binary`` is the paper-faithful objective: grade 1/2 is positive and
      grade 0 is negative.
    * ``graded`` preserves our 2/1/0 supervision as probabilities 1/.5/0.
    * ``relevance`` consumes the Judge relevance value in [0, 1].

    Rows with a negative grade (our ``uncertain`` sentinel) are ignored.
    ``sample_weight`` affects only the loss and is never a model input.
    """

    def __init__(self, target_mode: TargetMode = "binary") -> None:
        super().__init__()
        if target_mode not in {"binary", "graded", "relevance"}:
            raise ValueError(f"unsupported target_mode: {target_mode}")
        self.target_mode = target_mode

    def forward(
        self,
        logits: Tensor,
        target_grade: Tensor,
        *,
        target_relevance: Tensor | None = None,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        if logits.shape != target_grade.shape:
            raise ValueError("logits and target_grade must have identical shapes")
        valid = target_grade >= 0
        if not torch.any(valid):
            # Keep the zero connected to logits so backward() is still valid.
            return logits.sum() * 0.0

        valid_logits = logits[valid]
        valid_grades = target_grade[valid]
        if self.target_mode == "binary":
            targets = (valid_grades > 0).to(dtype=valid_logits.dtype)
        elif self.target_mode == "graded":
            targets = valid_grades.to(dtype=valid_logits.dtype) / 2.0
        else:
            if target_relevance is None or target_relevance.shape != target_grade.shape:
                raise ValueError(
                    "target_relevance with the same shape is required in relevance mode"
                )
            targets = target_relevance[valid].to(dtype=valid_logits.dtype)
            if torch.any((targets < 0) | (targets > 1)):
                raise ValueError("target_relevance values must lie in [0, 1]")

        losses = F.binary_cross_entropy_with_logits(
            valid_logits, targets, reduction="none"
        )
        if sample_weight is None:
            return losses.mean()
        if sample_weight.shape != target_grade.shape:
            raise ValueError("sample_weight and target_grade must have identical shapes")
        weights = sample_weight[valid].to(dtype=losses.dtype)
        if torch.any(weights < 0):
            raise ValueError("sample weights must be non-negative")
        denominator = weights.sum()
        if denominator <= 0:
            return losses.sum() * 0.0
        return (losses * weights).sum() / denominator
