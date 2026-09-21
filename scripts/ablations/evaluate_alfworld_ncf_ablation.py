#!/usr/bin/env python3
"""Evaluate one trained ALFWorld ablation checkpoint under the locked protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "scripts/ablations",
    ROOT / "scripts/memp",
    ROOT / "src/GraphOfSkills_NCF",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_memory_query_only_candidates import signature, view_relation  # noqa: E402
from gos.ncf.models import GMF, MLP, NCFConfig, NeuMF  # noqa: E402
from replay_alfworld_ncf_protocols import (  # noqa: E402
    memory_metrics,
    normalized,
    rank_descending,
    read_json,
    read_jsonl,
    score_matrix,
    sha256_file,
    skill_metrics,
    write_json,
    write_jsonl,
)
from train_alfworld_ncf_ablation import LinearScorer  # noqa: E402


DEFAULT_MANIFEST = ROOT / "configs/ablations/alfworld_ncf_v1.json"


def load_checkpoint(path: Path, model_type: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if model_type == "linear":
        model = LinearScorer(int(checkpoint["model_config"]["input_dim"]))
    else:
        classes = {"gmf": GMF, "mlp": MLP, "neumf": NeuMF}
        model = classes[model_type](NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def skill_inputs(spec: dict[str, Any]):
    reference = read_json(ROOT / spec["reference_report"]["path"])
    rows = reference["rows"]
    skill_cache = read_json(ROOT / spec["graph_embeddings"]["path"])
    eval_cache = read_json(ROOT / spec["evaluation_embeddings"]["path"])["entries"]
    skill_ids = sorted(skill_cache)

    def embedding(text: str) -> np.ndarray:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entry = eval_cache.get(key)
        if entry is None or entry.get("text") != text:
            raise KeyError(f"missing frozen evaluation embedding: {text!r}")
        return np.asarray(entry["embedding"], dtype=np.float32)

    local_queries = np.stack([embedding(row["query"]) for row in rows])
    contexts = np.stack([embedding(row["context"]) for row in rows])
    resources = np.stack(
        [np.asarray(skill_cache[skill_id]["embedding"], dtype=np.float32) for skill_id in skill_ids]
    )
    cosine_scores = normalized(local_queries) @ normalized(resources).T
    cosine_orders = rank_descending(cosine_scores)
    return rows, skill_ids, contexts, resources, cosine_orders


def evaluate_skill(
    spec: dict[str, Any], model: torch.nn.Module, batch_size: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, skill_ids, contexts, resources, cosine_orders = skill_inputs(spec)
    scores = score_matrix(model, contexts, resources, batch_size)
    rankings: list[list[str]] = []
    output_rows = []
    for query_index, row in enumerate(rows):
        candidates = cosine_orders[query_index, :12]
        order = candidates[np.argsort(-scores[query_index, candidates], kind="stable")]
        ranking = [skill_ids[int(index)] for index in order]
        rankings.append(ranking)
        output_rows.append(
            {
                "query_index": query_index,
                "task_id": row["task_id"],
                "phase": row["phase"],
                "gold": row["gold"],
                "ranking": ranking,
            }
        )
    metrics = skill_metrics(rankings, [set(row["gold"]) for row in rows], 5)
    return metrics, output_rows


def memory_inputs(spec: dict[str, Any]):
    views = read_jsonl(ROOT / spec["query_views"]["path"])
    memories = read_jsonl(ROOT / spec["memory_bank"]["path"])
    with np.load(ROOT / spec["arrays"]["path"], allow_pickle=False) as payload:
        task_ids = payload["task_ids"].copy()
        all_queries = payload["task_embeddings"].copy()
        resources = payload["memory_embeddings"].copy()
    positions = {int(task_id): index for index, task_id in enumerate(task_ids)}
    selected = [
        row for row in views
        if row["model_split"] == "model_dev" and int(row["query_view_index"]) in positions
    ]
    selected.sort(key=lambda row: int(row["query_view_index"]))
    queries = all_queries[[positions[int(row["query_view_index"])] for row in selected]]
    grades = np.zeros((len(selected), len(memories)), dtype=np.int8)
    resource_signatures = [signature(row["signature"]) for row in memories]
    for query_index, view in enumerate(selected):
        query_signature = signature(view["signature"])
        for resource_index, (memory, resource_signature) in enumerate(
            zip(memories, resource_signatures, strict=True)
        ):
            grade = 0
            if bool(memory["procedurally_valid"]):
                grade, _ = view_relation(view["view_type"], query_signature, resource_signature)
            grades[query_index, resource_index] = int(grade)
    return selected, queries, resources, grades


def evaluate_memory(
    spec: dict[str, Any], model: torch.nn.Module, batch_size: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    views, queries, resources, grades = memory_inputs(spec)
    scores = score_matrix(model, queries, resources, batch_size)
    rankings = rank_descending(scores)
    metrics = memory_metrics(grades, rankings, 10)
    output_rows = []
    for query_index, view in enumerate(views):
        ranking = rankings[query_index].astype(int).tolist()
        output_rows.append(
            {
                "query_index": query_index,
                "query_view_index": int(view["query_view_index"]),
                "view_type": view["view_type"],
                "ranking": ranking,
                "top10_grades": grades[query_index, ranking[:10]].astype(int).tolist(),
            }
        )
    return metrics, output_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--domain", choices=("skill", "memory"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model-type", choices=("linear", "gmf", "mlp", "neumf"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    section = "skill_alfworld" if args.domain == "skill" else "memory_alfworld"
    model = load_checkpoint(args.checkpoint, args.model_type)
    if args.domain == "skill":
        metrics, rows = evaluate_skill(manifest[section], model, args.batch_size)
        protocol = "fixed cosine Top-12 candidates, scorer-only reranking, final K=5"
    else:
        metrics, rows = evaluate_memory(manifest[section], model, args.batch_size)
        protocol = "fixed full 300-Memory pool, scorer-only ranking, final K=10"
    report = {
        "schema_version": "agent_skill_evolution.alfworld_ncf_ablation_eval.v1",
        "domain": args.domain,
        "variant": args.variant,
        "model_type": args.model_type,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "protocol": protocol,
        "api_calls_made": 0,
        "metrics": metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "metrics.json", report)
    write_jsonl(args.output / "per_query_rankings.jsonl", rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
