#!/usr/bin/env python3
"""Evaluate the paper-table Memory retrievers on a full frozen Memory bank.

This is an internal, train-derived model-dev evaluation.  It never reads the
official ALFWorld Dev/Test tasks, trajectories, rewards, or agent outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts/memp", ROOT / "scripts/ncf_v3"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_memory_query_only_candidates import signature, view_relation  # noqa: E402
from phase_neumf_common import NCFConfig, NeuMF  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def dcg(grades: Iterable[int]) -> float:
    return sum((2**int(grade) - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def metrics_for_rankings(grades: np.ndarray, rankings: np.ndarray, final_k: int) -> dict[str, float | int]:
    """Compute the frozen table contract.

    Ret@K is a hit-rate over queries that have at least one Grade-2 Memory in
    the full bank.  MRR is truncated at ``final_k``.  Precision treats both
    Grade 2 and Grade 1 as compatible, while NDCG preserves the 2/1/0 gains.
    """
    if grades.ndim != 2 or rankings.ndim != 2 or grades.shape[0] != rankings.shape[0]:
        raise ValueError("grades and rankings must have the same query dimension")
    if rankings.shape[1] < final_k:
        raise ValueError("ranking contains fewer items than final-k")
    if np.any(rankings < 0) or np.any(rankings >= grades.shape[1]):
        raise ValueError("ranking contains an out-of-range candidate index")
    eligible = np.any(grades == 2, axis=1)
    if not np.any(eligible):
        raise ValueError("evaluation contains no Grade-2-eligible query")

    hit_values: dict[int, list[float]] = {1: [], 5: [], 10: []}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    precisions: list[float] = []
    for query_index in range(grades.shape[0]):
        order = rankings[query_index, :final_k]
        ranked_grades = grades[query_index, order]
        if eligible[query_index]:
            for k in hit_values:
                hit_values[k].append(float(np.any(ranked_grades[: min(k, final_k)] == 2)))
            first = np.flatnonzero(ranked_grades == 2)
            reciprocal_ranks.append(1.0 / (int(first[0]) + 1) if len(first) else 0.0)
        ideal = np.sort(grades[query_index])[::-1][:final_k]
        denominator = dcg(ideal)
        ndcgs.append(dcg(ranked_grades) / denominator if denominator else 0.0)
        precisions.append(float(np.mean(ranked_grades > 0)))

    return {
        "query_count": int(grades.shape[0]),
        "grade_2_eligible_query_count": int(np.sum(eligible)),
        "ret@1": float(np.mean(hit_values[1])),
        "ret@5": float(np.mean(hit_values[5])),
        "ret@10": float(np.mean(hit_values[10])),
        "mrr@10": float(np.mean(reciprocal_ranks)),
        "ndcg@10": float(np.mean(ndcgs)),
        "compatible_precision@10": float(np.mean(precisions)),
    }


@torch.no_grad()
def score_full_bank(
    model: NeuMF,
    task_embeddings: np.ndarray,
    memory_embeddings: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    query_count, memory_count = len(task_embeddings), len(memory_embeddings)
    task_indices = np.repeat(np.arange(query_count), memory_count)
    memory_indices = np.tile(np.arange(memory_count), query_count)
    scores: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(task_indices), batch_size):
        selected = slice(start, start + batch_size)
        task = torch.from_numpy(task_embeddings[task_indices[selected]]).float()
        memory = torch.from_numpy(memory_embeddings[memory_indices[selected]]).float()
        scores.append(model(task, memory).cpu().numpy())
    return np.concatenate(scores).reshape(query_count, memory_count)


def rank_descending(scores: np.ndarray) -> np.ndarray:
    # Stable sort makes Memory ID the deterministic tie breaker.
    return np.argsort(-scores, axis=1, kind="stable")


def cosine_then_neumf(cosine: np.ndarray, neumf: np.ndarray, candidate_c: int) -> np.ndarray:
    cosine_order = rank_descending(cosine)[:, :candidate_c]
    output = np.empty_like(cosine_order)
    for query_index, candidates in enumerate(cosine_order):
        rerank = np.argsort(-neumf[query_index, candidates], kind="stable")
        output[query_index] = candidates[rerank]
    return output


def mean_metric_dict(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = values[0] if key.endswith("count") else float(np.mean(values))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-views", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-c", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    if not 1 <= args.final_k <= args.candidate_c:
        raise ValueError("require 1 <= final-k <= candidate-c")

    views = load_jsonl(args.query_views)
    memories = load_jsonl(args.memories)
    if len(memories) != 300:
        raise ValueError("the frozen Script Memory bank must contain 300 entries")
    if any(int(row["memory_id"]) != index for index, row in enumerate(memories)):
        raise ValueError("Memory IDs must match frozen bank ordering")
    if any(bool(row.get("uses_eval_data")) for row in views):
        raise ValueError("official evaluation data entered the internal retrieval table")

    with np.load(args.arrays, allow_pickle=False) as payload:
        task_ids = payload["task_ids"].copy()
        all_task_embeddings = payload["task_embeddings"].copy()
        memory_embeddings = payload["memory_embeddings"].copy()
    task_index_by_id = {int(task_id): index for index, task_id in enumerate(task_ids)}
    selected_views = [
        row
        for row in views
        if row["model_split"] == "model_dev"
        and int(row["query_view_index"]) in task_index_by_id
    ]
    selected_views.sort(key=lambda row: int(row["query_view_index"]))
    selected_task_indices = [task_index_by_id[int(row["query_view_index"])] for row in selected_views]
    task_embeddings = all_task_embeddings[selected_task_indices]

    grades = np.zeros((len(selected_views), len(memories)), dtype=np.int8)
    memory_signatures = [signature(row["signature"]) for row in memories]
    for query_index, view in enumerate(selected_views):
        task_signature = signature(view["signature"])
        for memory_index, (memory, memory_signature) in enumerate(zip(memories, memory_signatures)):
            if not bool(memory["procedurally_valid"]):
                grade = 0
            else:
                grade, _ = view_relation(view["view_type"], task_signature, memory_signature)
            grades[query_index, memory_index] = int(grade)

    checkpoint = torch.load(args.model_dir / "neumf.pt", map_location="cpu", weights_only=True)
    model = NeuMF(NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    cosine = normalized(task_embeddings) @ normalized(memory_embeddings).T
    neumf = score_full_bank(model, task_embeddings, memory_embeddings, args.batch_size)

    rankings = {
        "query_cosine": rank_descending(cosine),
        "content_neumf": rank_descending(neumf),
        "cosine_then_content_neumf": cosine_then_neumf(cosine, neumf, args.candidate_c),
    }
    method_metrics = {
        name: metrics_for_rankings(grades, ranking, args.final_k)
        for name, ranking in rankings.items()
    }
    rng = np.random.default_rng(args.seed)
    random_rows = []
    base = np.arange(len(memories))
    for _ in range(args.random_repeats):
        random_ranking = np.stack([rng.permutation(base) for _ in selected_views])
        random_rows.append(metrics_for_rankings(grades, random_ranking, args.final_k))
    method_metrics["random"] = mean_metric_dict(random_rows)

    grade_counts = {str(grade): int(np.sum(grades == grade)) for grade in (0, 1, 2)}
    report = {
        "schema_version": "task_resource_ncf.memory_full_bank_retrieval_table.v1",
        "evaluation_split": "train-derived internal model-dev",
        "official_dev_or_test_read": False,
        "memory_count": len(memories),
        "query_count": len(selected_views),
        "candidate_c": args.candidate_c,
        "final_k": args.final_k,
        "random_repeats": args.random_repeats,
        "grade_counts": grade_counts,
        "metric_contract": {
            "ret@k": "Grade-2 hit rate among queries with at least one full-bank Grade-2 Memory",
            "mrr@10": "reciprocal rank of first Grade-2 Memory, zero when absent from final Top-10",
            "ndcg@10": "graded 2/1/0 relevance over final Top-10",
            "compatible_precision@10": "fraction of final Top-10 with Grade 2 or Grade 1",
        },
        "methods": method_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
