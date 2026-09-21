#!/usr/bin/env python3
"""Aggregate atomic operation scores back to the fixed 45-Memory interface."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
NCF_SRC = ROOT / "src/GraphOfSkills_NCF"
if str(NCF_SRC) not in sys.path:
    sys.path.insert(0, str(NCF_SRC))
from gos.ncf.models import NCFConfig, NeuMF  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dcg(values: list[float]) -> float:
    return sum((2**value - 1) / math.log2(rank + 2) for rank, value in enumerate(values))


def rank_metrics(scores: np.ndarray, relevance: np.ndarray) -> dict[str, float]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    ideal = np.sort(relevance)[::-1]
    best = float(np.max(relevance))
    best_ids = set(np.flatnonzero(np.isclose(relevance, best)).tolist())
    first_best = next((rank for rank, item in enumerate(order, 1) if int(item) in best_ids), None)
    output: dict[str, float] = {
        "ret@1": float(any(int(item) in best_ids for item in order[:1])),
        "ret@3": float(any(int(item) in best_ids for item in order[:3])),
        "ret@5": float(any(int(item) in best_ids for item in order[:5])),
        "mrr": 1.0 / first_best if first_best else 0.0,
    }
    for k in (3, 5, 10):
        actual = relevance[order[:k]].tolist()
        output[f"ndcg@{k}"] = dcg(actual) / max(dcg(ideal[:k].tolist()), 1e-12)
    return output


def evaluate_budget(
    *,
    budget: int,
    dev_subgoals_by_family: dict[str, list[str]],
    subgoal_index: dict[str, int],
    task_embeddings: np.ndarray,
    operation_norm: np.ndarray,
    parent_ids: np.ndarray,
    evidence: dict[tuple[str, str, int], list[int]],
    neumf_by_subgoal: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    unique_parents = np.arange(45)
    metric_rows: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
    flips: defaultdict[str, int] = defaultdict(int)
    for family_id, ids in sorted(dev_subgoals_by_family.items()):
        cosine_parent_subgoals, neumf_parent_subgoals, gold_parent_subgoals = [], [], []
        for subgoal_id in ids:
            query = task_embeddings[subgoal_index[subgoal_id]].astype(np.float32)
            query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
            cosine_operation = operation_norm @ query_norm
            logits = neumf_by_subgoal[subgoal_id]
            candidates = np.argsort(-cosine_operation, kind="stable")[:budget]
            candidate_mask = np.zeros(len(parent_ids), dtype=bool)
            candidate_mask[candidates] = True
            cosine_parent = np.zeros(45, dtype=np.float32)
            neumf_parent = np.zeros(45, dtype=np.float32)
            gold_parent = np.zeros(45, dtype=np.float32)
            for parent_id in unique_parents:
                mask = (parent_ids == parent_id) & candidate_mask
                if np.any(mask):
                    cosine_parent[parent_id] = max(float(np.max(cosine_operation[mask])), 0.0)
                    # NeuMF target is Grade/2, so sigmoid produces the natural
                    # non-negative utility used by missing-subgoal coverage.
                    neumf_parent[parent_id] = float(1.0 / (1.0 + np.exp(-np.max(logits[mask]))))
                grades = evidence.get((family_id, subgoal_id, int(parent_id)), [])
                gold_parent[parent_id] = max(grades, default=0) / 2.0
            cosine_parent_subgoals.append(cosine_parent)
            neumf_parent_subgoals.append(neumf_parent)
            gold_parent_subgoals.append(gold_parent)
        cosine_scores = np.mean(cosine_parent_subgoals, axis=0)
        neumf_scores = np.mean(neumf_parent_subgoals, axis=0)
        gold_relevance = np.mean(gold_parent_subgoals, axis=0)
        left = rank_metrics(cosine_scores, gold_relevance)
        right = rank_metrics(neumf_scores, gold_relevance)
        metric_rows["cosine"].append(left); metric_rows["content_neumf"].append(right)
        l1, r1 = left["ret@1"] == 1, right["ret@1"] == 1
        flips["both_success" if l1 and r1 else "cosine_only" if l1 else "neumf_only" if r1 else "both_fail"] += 1
    summary = {
        method: {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("ret@1", "ret@3", "ret@5", "mrr", "ndcg@3", "ndcg@5", "ndcg@10")
        }
        for method, rows in metric_rows.items()
    }
    return summary, dict(flips)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subgoals", type=Path, required=True)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--dev-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    subgoals = read_jsonl(args.subgoals)
    operations = read_jsonl(args.operations)
    dev_pairs = read_jsonl(args.dev_pairs)
    if any(row.get("validation_or_test_used") is not False for row in subgoals + operations + dev_pairs):
        raise ValueError("evaluation-derived input detected")
    with np.load(args.arrays, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NeuMF(NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    subgoal_index = {str(value): index for index, value in enumerate(arrays["subgoal_ids"])}
    operation_index = {str(value): index for index, value in enumerate(arrays["operation_ids"])}
    parent_ids = arrays["operation_parent_memory_ids"].astype(int)
    unique_parents = np.arange(int(np.max(parent_ids)) + 1)
    if not np.array_equal(unique_parents, np.arange(45)):
        raise ValueError("expected the fixed 45-parent Memory interface")

    dev_subgoals_by_family: defaultdict[str, list[str]] = defaultdict(list)
    for row in subgoals:
        if row["model_split"] == "internal_dev":
            dev_subgoals_by_family[str(row["family_id"])].append(str(row["subgoal_id"]))
    evidence: defaultdict[tuple[str, str, int], list[int]] = defaultdict(list)
    for row in dev_pairs:
        operation = operations[operation_index[str(row["operation_id"])]]
        evidence[(str(row["family_id"]), str(row["subgoal_id"]), int(operation["parent_memory_id"]))].append(int(row["final_grade"]))

    operation_embeddings = arrays["memory_embeddings"].astype(np.float32)
    operation_norm = operation_embeddings / np.maximum(np.linalg.norm(operation_embeddings, axis=1, keepdims=True), 1e-12)
    dev_subgoal_ids = [subgoal_id for _, ids in sorted(dev_subgoals_by_family.items()) for subgoal_id in ids]
    dev_query_indices = np.asarray([subgoal_index[value] for value in dev_subgoal_ids], dtype=np.int32)
    dev_queries = arrays["task_embeddings"][dev_query_indices].astype(np.float32)
    neumf_by_subgoal: dict[str, np.ndarray] = {}
    # Score small query blocks against all 519 operations.  This preserves the
    # exact Cartesian scoring while avoiding one Python model call per query.
    with torch.no_grad():
        for start in range(0, len(dev_queries), 32):
            query_block = dev_queries[start:start + 32]
            query_batch = np.repeat(query_block, len(operations), axis=0)
            operation_batch = np.tile(operation_embeddings, (len(query_block), 1))
            scores = model(torch.from_numpy(query_batch), torch.from_numpy(operation_batch)).numpy()
            scores = scores.reshape(len(query_block), len(operations))
            for offset, row in enumerate(scores):
                neumf_by_subgoal[dev_subgoal_ids[start + offset]] = row
    budgets = {}
    for budget in (5, 10, 20, 50, 519):
        summaries, flips = evaluate_budget(
            budget=budget,
            dev_subgoals_by_family=dev_subgoals_by_family,
            subgoal_index=subgoal_index,
            task_embeddings=arrays["task_embeddings"],
            operation_norm=operation_norm,
            parent_ids=parent_ids,
            evidence=evidence,
            neumf_by_subgoal=neumf_by_subgoal,
        )
        budgets[str(budget)] = {
            **summaries,
            "delta_neumf_minus_cosine": {
                key: summaries["content_neumf"][key] - summaries["cosine"][key]
                for key in summaries["cosine"]
            },
            "ret1_paired_flips": flips,
        }
    report: dict[str, Any] = {
        "schema_version": "memp.travelplanner.atomic_parent_retrieval_eval.v1",
        "task_family_count": len(dev_subgoals_by_family),
        "parent_memory_count": 45,
        "aggregation": "max operation per parent per subgoal, then mean across task subgoals",
        "gold": "same aggregation over resolved internal-dev Grade/2 evidence",
        "retrieval": "operation cosine Top-C, followed by direct NeuMF ranking; no score fusion",
        "candidate_budget_results": budgets,
        "official_validation_or_test_used": False,
        "caveat": "Train-derived internal family-dev parent-Memory diagnostic; not official #CS/#HC.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
