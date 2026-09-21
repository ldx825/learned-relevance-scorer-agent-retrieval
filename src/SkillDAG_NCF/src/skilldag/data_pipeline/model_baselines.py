"""Unified offline baselines for the leakage-aware model data."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .action_candidates import _atomic_json, _load_jsonl


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _metrics(ranked: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, float]:
    judged = [row for row in ranked if row["grade"] >= 0]
    ideal = sorted((row["grade"] for row in judged), reverse=True)
    required_total = sum(row["grade"] == 2 for row in judged)
    first_required = next((i for i, row in enumerate(judged, 1) if row["grade"] == 2), None)
    result = {"mrr_required": 1 / first_required if first_required else 0.0}
    for k in ks:
        top = judged[:k]
        ideal_dcg = _dcg(ideal[:k])
        result[f"ndcg@{k}"] = _dcg([row["grade"] for row in top]) / ideal_dcg if ideal_dcg else 0.0
        result[f"required_recall@{k}"] = (
            sum(row["grade"] == 2 for row in top) / required_total if required_total else 0.0
        )
        result[f"bad_item_rate@{k}"] = (
            sum(row["grade"] == 0 for row in top) / len(top) if top else 0.0
        )
        result[f"harmful_item_rate@{k}"] = (
            sum(row["is_harmful"] for row in top) / len(top) if top else 0.0
        )
        ids = [row["skill_id"] for row in top]
        redundant = sum(
            any(frozenset({skill_id, prior}) in row["similar_pairs"] for prior in ids[:index])
            for index, (skill_id, row) in enumerate(zip(ids, top))
        )
        result[f"redundant_item_rate@{k}"] = redundant / len(top) if top else 0.0
    return result


def evaluate_model_baselines(
    dataset_dir: Path | str,
    arrays_path: Path | str,
    output_root: Path | str,
    *,
    splits: tuple[str, ...] = ("dev", "test"),
    ks: tuple[int, ...] = (3, 5, 10),
    similar_penalty: float = 0.05,
) -> dict[str, Any]:
    """Compare deployable cosine/graph baselines with privileged RRF."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    arrays_path = Path(arrays_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    pairs = _load_jsonl(dataset_dir / "pairs.jsonl")
    with np.load(arrays_path, allow_pickle=False) as data:
        pair_ids = [str(value) for value in data["pair_ids"]]
        cosine = data["pair_features"][:, 0].astype(float)
    if len(pair_ids) != len(pairs) or any(pair_ids[i] != row["pair_id"] for i, row in enumerate(pairs)):
        raise ValueError("model arrays and dataset pairs are not aligned")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        similar_pairs = {
            frozenset({pair["skill_id"], relation["skill_id"]})
            for relation in pair.get("graph_relations_to_candidates", [])
            if relation.get("type") == "similar_to"
        }
        by_task[pair["task_record_id"]].append(
            {
                "skill_id": pair["skill_id"],
                "split": pair["dataset_split"],
                "grade": -1 if pair["target_grade"] is None else int(pair["target_grade"]),
                "is_harmful": bool(pair["is_harmful"]),
                "cosine": cosine[index],
                "task_rank": pair.get("task_semantic_rank"),
                "action_rank": pair.get("action_semantic_rank"),
                "similar_pairs": similar_pairs,
            }
        )

    methods = {
        "student_cosine": {"deployable": True, "uses_privileged_features": False},
        "student_cosine_graph_soft": {"deployable": True, "uses_privileged_features": False},
        "privileged_rrf_diagnostic": {"deployable": False, "uses_privileged_features": True},
    }
    subset_reports: dict[str, Any] = {}
    for split in splits:
        task_ids = sorted(task_id for task_id, rows in by_task.items() if rows[0]["split"] == split)
        method_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
        for task_id in task_ids:
            rows = by_task[task_id]
            cosine_ranked = sorted(rows, key=lambda row: (-row["cosine"], row["skill_id"]))
            method_metrics["student_cosine"].append(_metrics(cosine_ranked, ks))

            remaining = list(cosine_ranked)
            graph_ranked: list[dict[str, Any]] = []
            selected_ids: list[str] = []
            while remaining:
                best = max(
                    remaining,
                    key=lambda row: (
                        row["cosine"]
                        - similar_penalty
                        * sum(
                            frozenset({row["skill_id"], prior}) in row["similar_pairs"]
                            for prior in selected_ids
                        ),
                        -ord(row["skill_id"][0]),
                    ),
                )
                remaining.remove(best)
                graph_ranked.append(best)
                selected_ids.append(best["skill_id"])
            method_metrics["student_cosine_graph_soft"].append(_metrics(graph_ranked, ks))

            def rrf(row: dict[str, Any]) -> float:
                return (1 / (60 + int(row["task_rank"])) if row["task_rank"] else 0.0) + (
                    1 / (60 + int(row["action_rank"])) if row["action_rank"] else 0.0
                )

            privileged = sorted(rows, key=lambda row: (-rrf(row), row["skill_id"]))
            method_metrics["privileged_rrf_diagnostic"].append(_metrics(privileged, ks))
        subset_reports[split] = {
            "task_count": len(task_ids),
            "methods": {
                method: {
                    metric: sum(row[metric] for row in values) / len(values)
                    for metric in sorted(values[0])
                }
                for method, values in method_metrics.items()
            },
        }

    report = {
        "schema_version": "skilldag_ncf.model_baselines.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_made": 0,
        "trained_models": 0,
        "candidate_pool": "Judge-labeled task/action union; selection is partly privileged",
        "methods": methods,
        "similar_penalty": similar_penalty,
        "ks": list(ks),
        "subsets": subset_reports,
        "limitations": [
            "The candidate pool contains action-derived candidates and is not deployable as-is.",
            "privileged_rrf_diagnostic uses expert-plan-derived action ranks and is not a student baseline.",
            "Metrics measure agreement with Judge weak labels, not ALFWorld execution success.",
        ],
    }
    report_path = output_root / "benchmarks" / "task_skill_v1_baselines" / "report.json"
    report["outputs"] = {"report": str(report_path)}
    _atomic_json(report_path, report)
    return report
