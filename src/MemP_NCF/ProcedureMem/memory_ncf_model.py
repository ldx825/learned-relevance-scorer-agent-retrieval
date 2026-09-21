"""NeuMF with explicit, runtime-observable task-memory relation features."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from gos.ncf.models import NCFConfig, NeuMF


PAIR_FEATURE_NAMES = (
    "same_operation",
    "same_object",
    "same_destination",
    "same_cardinality",
    "task_is_transformation",
    "memory_is_plain_placement",
    "task_is_pick_two",
    "memory_is_pick_place",
    "memory_procedurally_valid",
    "same_operation_and_object",
    "same_operation_and_destination",
    "same_object_and_destination",
    "same_operation_object_destination",
    "transformation_placement_same_object_destination",
    "pick_two_placement_same_object_destination",
)


def relation_features(
    task_signature: dict[str, Any],
    memory_signature: dict[str, Any],
    memory_procedurally_valid: bool,
) -> np.ndarray:
    task_operation = str(task_signature["operation"])
    memory_operation = str(memory_signature["operation"])
    same_operation = task_operation == memory_operation
    same_object = str(task_signature["object_type"]) == str(memory_signature["object_type"])
    same_destination = str(task_signature["destination"]) == str(
        memory_signature["destination"]
    )
    task_is_transformation = task_operation in {"clean", "heat", "cool"}
    memory_is_pick_place = memory_operation == "pick_place"
    task_is_pick_two = task_operation == "pick_two"
    return np.asarray(
        [
            same_operation,
            same_object,
            same_destination,
            int(task_signature["cardinality"]) == int(memory_signature["cardinality"]),
            task_is_transformation,
            memory_is_pick_place,
            task_is_pick_two,
            memory_is_pick_place,
            memory_procedurally_valid,
            same_operation and same_object,
            same_operation and same_destination,
            same_object and same_destination,
            same_operation and same_object and same_destination,
            task_is_transformation
            and memory_is_pick_place
            and same_object
            and same_destination,
            task_is_pick_two
            and memory_is_pick_place
            and same_object
            and same_destination,
        ],
        dtype=np.float32,
    )


class RelationNeuMF(nn.Module):
    """Standard content NeuMF plus a learned head over observable relations."""

    def __init__(self, config: NCFConfig, pair_feature_dim: int) -> None:
        super().__init__()
        self.neumf = NeuMF(config)
        self.relation_head = nn.Linear(pair_feature_dim, 1, bias=False)
        nn.init.zeros_(self.relation_head.weight)

    def forward(
        self,
        task_embedding: torch.Tensor,
        memory_embedding: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.neumf(task_embedding, memory_embedding) + self.relation_head(
            pair_features
        ).squeeze(-1)
