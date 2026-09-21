"""Dependency-free NeuMF inference for benchmark containers.

SkillsBench task images are intentionally heterogeneous and cannot be assumed
to contain PyTorch.  Training and export still use PyTorch on the host, while
this module consumes a JSON inference bundle with:

* task-side projection matrices;
* precomputed skill-side projection vectors for the fixed skill library;
* the small MLP tower and output layer.

The computation is algebraically identical to ``NeuMFReranker``.  Only the
skill-side linear projections are moved to export time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "skilldag.portable_neumf.v1"


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]


def _linear(
    vector: list[float],
    weight: list[list[float]],
    bias: list[float],
) -> list[float]:
    return [
        value + offset
        for value, offset in zip(_matvec(weight, vector), bias)
    ]


class PortableNeuMFReranker:
    """Run one exported content-NeuMF model using only the Python stdlib."""

    def __init__(self, bundle_path: str | Path, *, input_dim: int) -> None:
        self.bundle_path = Path(bundle_path).expanduser().resolve()
        payload = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported portable NeuMF schema: {payload.get('schema_version')!r}"
            )
        model_config = payload.get("model_config") or {}
        expected_dim = int(model_config.get("input_dim", 0))
        if expected_dim != input_dim:
            raise ValueError(
                f"portable checkpoint expects embedding dimension {expected_dim}, "
                f"got {input_dim}"
            )

        task = payload.get("task_projections") or {}
        self.gmf_task_weight = task.get("gmf_weight") or []
        self.mlp_task_weight = task.get("mlp_weight") or []
        self.skill_features = payload.get("skill_features") or {}
        self.mlp_layers = payload.get("mlp_layers") or []
        self.output = payload.get("output") or {}
        self._validate(input_dim)

    def _validate(self, input_dim: int) -> None:
        for name, matrix in (
            ("gmf task projection", self.gmf_task_weight),
            ("mlp task projection", self.mlp_task_weight),
        ):
            if not matrix or any(len(row) != input_dim for row in matrix):
                raise ValueError(f"invalid {name} in {self.bundle_path}")
        if not self.skill_features:
            raise ValueError(f"portable bundle has no skill features: {self.bundle_path}")
        if not self.mlp_layers:
            raise ValueError(f"portable bundle has no MLP layers: {self.bundle_path}")
        if not self.output.get("weight") or "bias" not in self.output:
            raise ValueError(f"portable bundle has no output layer: {self.bundle_path}")

    @property
    def supported_skill_ids(self) -> set[str]:
        return set(self.skill_features)

    def score_by_id(
        self,
        task_embedding: list[float],
        skill_ids: list[str],
    ) -> list[float]:
        missing = [skill_id for skill_id in skill_ids if skill_id not in self.skill_features]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"portable bundle is missing skill features: {preview}")

        task_gmf = _matvec(self.gmf_task_weight, task_embedding)
        task_mlp = _matvec(self.mlp_task_weight, task_embedding)
        scores: list[float] = []
        for skill_id in skill_ids:
            features: dict[str, Any] = self.skill_features[skill_id]
            skill_gmf = features["gmf"]
            skill_mlp = features["mlp"]
            gmf = [left * right for left, right in zip(task_gmf, skill_gmf)]
            hidden = task_mlp + skill_mlp
            for layer in self.mlp_layers:
                hidden = _linear(hidden, layer["weight"], layer["bias"])
                hidden = [max(0.0, value) for value in hidden]
            output_weight = self.output["weight"]
            score = sum(weight * value for weight, value in zip(output_weight, gmf + hidden))
            score += float(self.output["bias"])
            scores.append(float(score))
        return scores
