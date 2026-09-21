#!/usr/bin/env python3
"""Calibrate GMF/MLP on train/dev and replay the frozen SkillDAG pipeline.

No official evaluation query contributes to priors or alpha selection.  The
frozen 501-query replay is read only after calibration has been fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "scripts/ablations",
    ROOT / "scripts/ncf_v3",
    ROOT / "src/SkillDAG_NCF/src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_alfworld_ncf_ablation import load_checkpoint, skill_inputs  # noqa: E402
from phase_neumf_common import (  # noqa: E402
    PhaseData,
    cosine_scores,
    model_scores,
    ranking_metrics,
    utc_now,
)
from replay_alfworld_ncf_protocols import (  # noqa: E402
    read_json,
    score_matrix,
    sha256_file,
    skill_metrics,
    write_json,
    write_jsonl,
)
from skilldag.hybrid_selection import select_diverse  # noqa: E402


DEFAULT_MANIFEST = ROOT / "configs/ablations/alfworld_ncf_v1.json"
DEFAULT_OUTPUT = ROOT / ".runtime/ablations/alfworld_ncf_v1/deployed_head_replay"


def normalize(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return [0.0] * len(values) if std < 1e-8 else [(value - mean) / std for value in values]


def normalized_within_query(data: PhaseData, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.empty(len(indices), dtype=np.float64)
    queries = data.pair_phase_indices[indices]
    for query_index in np.unique(queries):
        positions = np.flatnonzero(queries == query_index)
        current = values[positions]
        result[positions] = (current - current.mean()) / (current.std() + 1e-8)
    return result


def calibrate(data: PhaseData, model) -> dict[str, Any]:
    train = data.indices("train")
    train_logits = model_scores(model, data, train, 512)
    sums = np.zeros(len(data.skill_ids), dtype=np.float64)
    counts = np.zeros(len(data.skill_ids), dtype=np.int64)
    for row_index, logit in zip(train, train_logits, strict=True):
        resource_index = int(data.pair_skill_indices[row_index])
        sums[resource_index] += float(logit)
        counts[resource_index] += 1
    priors = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    dev = data.indices("dev")
    dev_logits = model_scores(model, data, dev, 512)
    specific = dev_logits - priors[data.pair_skill_indices[dev]]
    cosine_z = normalized_within_query(data, dev, cosine_scores(data, dev))
    learned_z = normalized_within_query(data, dev, specific)
    trials: dict[str, dict[str, Any]] = {}
    best: tuple[tuple[float, float, float], float] | None = None
    for alpha in (0.25, 0.50, 0.75):
        metrics = ranking_metrics(data, dev, alpha * cosine_z + (1.0 - alpha) * learned_z)
        trials[str(alpha)] = metrics
        key = (metrics["ndcg@3"], metrics["required_recall@3"], -metrics["bad_item_rate@3"])
        if best is None or key > best[0]:
            best = (key, alpha)
    assert best is not None
    return {
        "alpha": best[1],
        "alpha_grid": [0.25, 0.50, 0.75],
        "dev_trials": trials,
        "priors": priors,
        "statistics_source": "internal train only",
        "selection_source": "internal dev only",
        "external_eval_read_during_calibration": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=("gmf", "mlp", "neumf"),
        default=("gmf", "mlp", "neumf"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    spec = manifest["skill_alfworld"]
    data = PhaseData(ROOT / spec["arrays"]["path"], "cpu")

    # Calibration is completed for every head before the external replay is loaded.
    calibrations: dict[str, tuple[Any, dict[str, Any]]] = {}
    for head in args.heads:
        checkpoint = ROOT / spec["frozen_checkpoints"][head]["path"]
        model = load_checkpoint(checkpoint, head)
        calibrations[head] = (model, calibrate(data, model))

    rows, skill_ids, contexts, resources, cosine_orders = skill_inputs(spec)
    evaluation_cache = read_json(ROOT / spec["evaluation_embeddings"]["path"])["entries"]
    local_queries = []
    for row in rows:
        key = hashlib.sha256(row["query"].encode("utf-8")).hexdigest()
        entry = evaluation_cache.get(key)
        if entry is None or entry.get("text") != row["query"]:
            raise KeyError(f"missing frozen local-query embedding: {row['query']!r}")
        local_queries.append(np.asarray(entry["embedding"], dtype=np.float32))
    local_queries_array = np.stack(local_queries)
    cosine_matrix = (
        local_queries_array / np.maximum(np.linalg.norm(local_queries_array, axis=1, keepdims=True), 1e-12)
    ) @ (
        resources / np.maximum(np.linalg.norm(resources, axis=1, keepdims=True), 1e-12)
    ).T
    graph = read_json(ROOT / spec["graph"]["path"])
    graph_nodes = graph["nodes"]
    graph_edges = graph.get("edges", [])
    golds = [set(row["gold"]) for row in rows]
    output: dict[str, Any] = {
        "schema_version": "agent_skill_evolution.skilldag_head_deployed_replay.v1",
        "generated_at": utc_now(),
        "api_calls_made": 0,
        "query_count": len(rows),
        "candidate_pool": len(skill_ids),
        "candidate_c": 12,
        "final_k": 5,
        "heads": {},
    }

    for head, (model, calibration) in calibrations.items():
        scores = score_matrix(model, contexts, resources, 1024)
        rankings: list[list[str]] = []
        detail_rows: list[dict[str, Any]] = []
        priors = calibration.pop("priors")
        calibration_payload = {
            "schema_version": "skilldag_ncf.v3.phase_online_calibration.v1",
            "generated_at": utc_now(),
            "api_calls_made": 0,
            **calibration,
            "skill_logit_prior": {
                str(skill_id): float(value)
                for skill_id, value in zip(data.skill_ids, priors, strict=True)
            },
        }
        write_json(args.output / f"{head}_calibration.json", calibration_payload)
        alpha = float(calibration["alpha"])
        for query_index, row in enumerate(rows):
            candidate_indices = cosine_orders[query_index, :12]
            candidate_ids = [skill_ids[int(index)] for index in candidate_indices]
            cosine_values = [float(cosine_matrix[query_index, int(index)]) for index in candidate_indices]
            logits = [float(scores[query_index, int(index)]) for index in candidate_indices]
            specific = [value - float(priors[int(index)]) for value, index in zip(logits, candidate_indices)]
            cosine_z = normalize(cosine_values)
            learned_z = normalize(specific)
            candidates = [
                {
                    "skill_id": skill_id,
                    "description": graph_nodes[skill_id].get("description", ""),
                    "cosine_score": cosine_value,
                    "ncf_logit": logit,
                    "base_score": alpha * cosine_value_z + (1.0 - alpha) * learned_value_z,
                }
                for skill_id, cosine_value, logit, cosine_value_z, learned_value_z in zip(
                    candidate_ids, cosine_values, logits, cosine_z, learned_z, strict=True
                )
            ]
            candidates.sort(key=lambda item: (-item["base_score"], item["skill_id"]))
            cosine_safe_ids = {skill_ids[int(index)] for index in cosine_orders[query_index, :3]}
            selected = select_diverse(
                candidates,
                edges=graph_edges,
                cosine_safe_ids=cosine_safe_ids,
                top_k=5,
            )
            ranking = [item["skill_id"] for item in selected]
            rankings.append(ranking)
            detail_rows.append({
                "query_index": query_index,
                "task_id": row["task_id"],
                "phase": row["phase"],
                "gold": row["gold"],
                "ranking": ranking,
            })
        checkpoint = ROOT / spec["frozen_checkpoints"][head]["path"]
        output["heads"][head] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "calibration": calibration,
            "calibration_file": str((args.output / f"{head}_calibration.json").resolve()),
            "metrics": skill_metrics(rankings, golds, 5),
        }
        write_jsonl(args.output / f"{head}_rankings.jsonl", detail_rows)

    write_json(args.output / "report.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
