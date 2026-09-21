#!/usr/bin/env python3
"""Export a content-NeuMF checkpoint for dependency-free SkillsBench inference.

The SkillsBench task images are heterogeneous and cannot be assumed to contain
PyTorch.  This exporter moves the fixed skill-side linear projections to the
host and writes the remaining NeuMF weights to JSON.  The task-side computation
and final score remain algebraically identical to the PyTorch model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "skilldag.portable_neumf.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_list(tensor) -> list:
    return tensor.detach().cpu().to(dtype=tensor.new_zeros(()).float().dtype).tolist()


def _load_embeddings(path: Path) -> tuple[dict[str, list[float]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", payload)
    if not isinstance(items, dict):
        raise ValueError("embedding cache must be a skill-id map or contain an items map")
    vectors: dict[str, list[float]] = {}
    model = str(payload.get("model", ""))
    for skill_id, entry in items.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("embedding"), list):
            raise ValueError(f"invalid embedding cache entry for {skill_id!r}")
        vectors[str(skill_id)] = [float(value) for value in entry["embedding"]]
        entry_model = str(entry.get("model", ""))
        if model and entry_model and entry_model != model:
            raise ValueError("embedding cache contains more than one model")
        model = model or entry_model
    if not vectors:
        raise ValueError("embedding cache is empty")
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent embedding dimensions: {sorted(dimensions)}")
    return vectors, model


def _load_graph_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError(f"graph has no nodes: {path}")
    return {str(skill_id) for skill_id in nodes}


def _linear_without_bias(weight, vector):
    # torch is deliberately imported by main(), not by the benchmark runtime.
    return weight @ vector


def export_bundle(
    *,
    checkpoint_path: Path,
    embeddings_path: Path,
    graph_path: Path,
    output_path: Path,
    calibration_path: Path | None = None,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain state_dict and model_config")
    state = checkpoint["state_dict"]
    config = dict(checkpoint.get("model_config") or {})
    input_dim = int(config.get("input_dim", 0))
    vectors, embedding_model = _load_embeddings(embeddings_path)
    graph_ids = _load_graph_ids(graph_path)
    if set(vectors) != graph_ids:
        missing = sorted(graph_ids - set(vectors))
        extra = sorted(set(vectors) - graph_ids)
        raise ValueError(
            "graph/embedding skill IDs differ; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    actual_dim = len(next(iter(vectors.values())))
    if input_dim != actual_dim:
        raise ValueError(
            f"checkpoint input_dim={input_dim}, embedding dimension={actual_dim}"
        )

    gmf_skill_weight = state["gmf_skill_projection.weight"].float()
    mlp_skill_weight = state["mlp_skill_projection.weight"].float()
    skill_features = {}
    for skill_id in sorted(vectors):
        vector = torch.tensor(vectors[skill_id], dtype=torch.float32)
        skill_features[skill_id] = {
            "gmf": _as_list(_linear_without_bias(gmf_skill_weight, vector)),
            "mlp": _as_list(_linear_without_bias(mlp_skill_weight, vector)),
        }

    mlp_layers = []
    layer_index = 0
    while f"mlp_tower.{layer_index}.weight" in state:
        mlp_layers.append(
            {
                "weight": _as_list(state[f"mlp_tower.{layer_index}.weight"].float()),
                "bias": _as_list(state[f"mlp_tower.{layer_index}.bias"].float()),
            }
        )
        layer_index += 2  # Linear, ReLU, Linear, ReLU, ...
    if not mlp_layers:
        raise ValueError("checkpoint contains no MLP tower layers")

    calibration: dict[str, Any] = {}
    if calibration_path is not None:
        raw = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration = {
            "alpha": float(raw.get("alpha", 0.25)),
            "skill_logit_prior": {
                str(key): float(value)
                for key, value in (raw.get("skill_logit_prior") or {}).items()
                if str(key) in graph_ids
            },
            "source_sha256": _sha256(calibration_path),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_config": {
            "input_dim": input_dim,
            "gmf_latent_dim": int(config.get("gmf_latent_dim", 16)),
            "mlp_layer_sizes": list(config.get("mlp_layer_sizes", (64, 32, 16))),
        },
        "embedding_model": embedding_model,
        "skill_count": len(skill_features),
        "task_projections": {
            "gmf_weight": _as_list(state["gmf_task_projection.weight"].float()),
            "mlp_weight": _as_list(state["mlp_task_projection.weight"].float()),
        },
        "skill_features": skill_features,
        "mlp_layers": mlp_layers,
        "output": {
            "weight": _as_list(state["output.weight"].float())[0],
            "bias": float(_as_list(state["output.bias"].float())[0]),
        },
        "calibration": calibration,
        "provenance": {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "embeddings_sha256": _sha256(embeddings_path),
            "graph_sha256": _sha256(graph_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_bundle(
        checkpoint_path=args.checkpoint.resolve(),
        embeddings_path=args.embeddings.resolve(),
        graph_path=args.graph.resolve(),
        output_path=args.output.resolve(),
        calibration_path=args.calibration.resolve() if args.calibration else None,
    )
    print(
        f"exported {payload['skill_count']} skills to {args.output} "
        f"(schema={payload['schema_version']})"
    )


if __name__ == "__main__":
    main()
