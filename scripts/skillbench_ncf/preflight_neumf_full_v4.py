#!/usr/bin/env python3
"""Run training-time load, loss, gradient, and deployment-input checks."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
NCF_V3 = ROOT / "scripts/ncf_v3"
SKILLDAG_SCRIPTS = ROOT / "src/SkillDAG_NCF/scripts"
SKILLDAG_SRC = ROOT / "src/SkillDAG_NCF/src"
for path in (NCF_V3, SKILLDAG_SCRIPTS, SKILLDAG_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase_neumf_common import (  # noqa: E402
    NCFConfig,
    NeuMF,
    PhaseData,
    preference_pairs,
    seed_all,
)
from export_portable_ncf import (  # noqa: E402
    _load_embeddings,
    _load_graph_ids,
    export_bundle,
)
from skilldag.portable_ncf import PortableNeuMFReranker  # noqa: E402


ARRAYS = ROOT / "data/skillbench_ncf/model_data/neumf_full_v4/arrays.npz"
SKILL_EMBEDDINGS = (
    ROOT / "data/skillbench_ncf/enriched_skill_embeddings_v3/hybrid_skill_embeddings.json"
)
GRAPH = ROOT / ".runtime/skillbench_ncf/sources/skillgraph_1000.json"
REPORT = ROOT / "artifacts/skillbench_ncf/manifests/neumf_full_v4_preflight_report.json"


def finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def main() -> int:
    seed_all(2026)
    data = PhaseData(ARRAYS, "cpu")
    if data.phase_embeddings.shape != (3_066, 3_072):
        raise AssertionError(f"unexpected query matrix: {data.phase_embeddings.shape}")
    if data.skill_embeddings.shape != (1_000, 3_072):
        raise AssertionError(f"unexpected skill matrix: {data.skill_embeddings.shape}")
    if len(data.pair_ids) != 51_911:
        raise AssertionError(f"unexpected pair count: {len(data.pair_ids)}")
    if set(map(int, data.split_codes.tolist())) != {0, 1}:
        raise AssertionError("arrays contain a split other than train/dev")

    preferences = preference_pairs(data)
    if len(preferences[0]) == 0 or not np.isfinite(preferences[2]).all():
        raise AssertionError("pairwise preference construction failed")
    model = NeuMF(
        NCFConfig(input_dim=3_072, gmf_latent_dim=16, mlp_layer_sizes=(64, 32, 16))
    )

    train_indices = data.indices("train")[:64]
    query, skill, target, weight = data.batch(train_indices)
    pointwise = (
        F.binary_cross_entropy_with_logits(
            model(query, skill), target, reduction="none"
        )
        * weight
    ).sum() / weight.sum()
    pointwise.backward()
    if not math.isfinite(float(pointwise.detach())) or not finite_gradients(model):
        raise AssertionError("pointwise loss or gradients are non-finite")
    model.zero_grad(set_to_none=True)

    better, worse, pair_weights = preferences
    hi = better[:64]
    lo = worse[:64]
    query_hi, skill_hi, _, _ = data.batch(hi)
    query_lo, skill_lo, _, _ = data.batch(lo)
    weight_t = torch.as_tensor(pair_weights[:64])
    pairwise = (
        F.softplus(-(model(query_hi, skill_hi) - model(query_lo, skill_lo)))
        * weight_t
    ).sum() / weight_t.sum()
    pairwise.backward()
    if not math.isfinite(float(pairwise.detach())) or not finite_gradients(model):
        raise AssertionError("pairwise loss or gradients are non-finite")

    vectors, embedding_model = _load_embeddings(SKILL_EMBEDDINGS)
    graph_ids = _load_graph_ids(GRAPH)
    if set(vectors) != graph_ids or len(vectors) != 1_000:
        raise AssertionError("portable export skill IDs differ from graph")
    if embedding_model != "text-embedding-3-large":
        raise AssertionError(f"unexpected deployment embedding model: {embedding_model}")
    if {len(vector) for vector in vectors.values()} != {3_072}:
        raise AssertionError("deployment skill embedding dimensions differ")
    if not np.allclose(
        data.skill_embeddings[0],
        np.asarray(vectors[str(data.skill_ids[0])], dtype=np.float32),
        rtol=0,
        atol=0,
    ):
        raise AssertionError("training and portable-export skill representations differ")

    # Exercise the complete deployment path with the same enhanced embeddings:
    # checkpoint -> portable JSON -> stdlib inference, then compare against
    # PyTorch logits for identical query/skill vectors.
    model.eval()
    candidate_ids = [str(value) for value in data.skill_ids[:3]]
    query_vector = data.phase_embeddings[0].astype(np.float32)
    candidate_indices = np.arange(3, dtype=np.int64)
    with torch.no_grad():
        query_batch = torch.from_numpy(
            np.repeat(query_vector[None, :], len(candidate_ids), axis=0)
        )
        skill_batch = torch.from_numpy(data.skill_embeddings[candidate_indices])
        torch_scores = model(query_batch, skill_batch).numpy()
    with tempfile.TemporaryDirectory(prefix="skillbench_v4_preflight_") as temp:
        temp_path = Path(temp)
        checkpoint = temp_path / "neumf.pt"
        bundle = temp_path / "ncf_bundle.json"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": asdict(model.config),
            },
            checkpoint,
        )
        export_bundle(
            checkpoint_path=checkpoint,
            embeddings_path=SKILL_EMBEDDINGS,
            graph_path=GRAPH,
            output_path=bundle,
        )
        portable = PortableNeuMFReranker(bundle, input_dim=3_072)
        portable_scores = np.asarray(
            portable.score_by_id(query_vector.tolist(), candidate_ids),
            dtype=np.float32,
        )
    max_export_error = float(np.max(np.abs(torch_scores - portable_scores)))
    if max_export_error > 1e-5:
        raise AssertionError(
            f"portable/PyTorch score mismatch: max_abs_error={max_export_error}"
        )

    result = {
        "schema_version": "skillbench_ncf.neumf_full_preflight.v4",
        "status": "pass",
        "formal_training_started": False,
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "checks": {
            "arrays_load_with_alfworld_v3_trainer": True,
            "pointwise_loss_finite": True,
            "pointwise_gradients_finite": True,
            "pairwise_preference_pairs": len(better),
            "pairwise_loss_finite": True,
            "pairwise_gradients_finite": True,
            "training_export_skill_representation_exact_match": True,
            "portable_export_accepts_items_schema": True,
            "portable_bundle_end_to_end_equivalent": True,
            "portable_max_abs_score_error": max_export_error,
            "graph_skill_ids_equal_export_skill_ids": True,
        },
        "counts": {
            "queries": len(data.phase_ids)
            if hasattr(data, "phase_ids")
            else len(data.task_ids),
            "skills": len(data.skill_ids),
            "pairs": len(data.pair_ids),
            "train_pairs": len(data.indices("train")),
            "dev_pairs": len(data.indices("dev")),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
