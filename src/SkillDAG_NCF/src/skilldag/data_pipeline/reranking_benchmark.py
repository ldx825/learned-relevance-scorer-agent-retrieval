"""Offline reranking baselines over the fixed Judge pilot evidence."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_candidates import _atomic_json, _atomic_jsonl, _load_jsonl, _sha256_bytes


GRADE = {"required": 2, "helpful": 1, "irrelevant": 0, "harmful": 0}
METHODS = (
    "task_cosine",
    "action_cosine",
    "cosine_max",
    "cosine_mean",
    "rrf",
    "rrf_similar_dedup",
)


def _score(row: dict[str, Any], method: str) -> float:
    task_score = row.get("task_cosine_score")
    action_score = row.get("action_cosine_score")
    task_value = float(task_score) if task_score is not None else -1.0
    action_value = float(action_score) if action_score is not None else -1.0
    if method == "task_cosine":
        return task_value
    if method == "action_cosine":
        return action_value
    if method == "cosine_max":
        return max(task_value, action_value)
    if method == "cosine_mean":
        # Missing views remain -1: this deliberately rewards agreement between
        # task and action retrieval rather than treating absence as zero evidence.
        return (task_value + action_value) / 2.0
    if method in {"rrf", "rrf_similar_dedup"}:
        task_rank = row.get("task_semantic_rank")
        action_rank = row.get("action_semantic_rank")
        return (1.0 / (60 + int(task_rank)) if task_rank is not None else 0.0) + (
            1.0 / (60 + int(action_rank)) if action_rank is not None else 0.0
        )
    raise ValueError(f"unknown method: {method}")


def _rank(
    rows: list[dict[str, Any]], method: str, similar_pairs: set[frozenset[str]]
) -> list[tuple[dict[str, Any], float]]:
    ranked = sorted(
        ((row, _score(row, method)) for row in rows),
        key=lambda item: (-item[1], item[0]["skill_id"]),
    )
    if method != "rrf_similar_dedup":
        return ranked
    selected: list[tuple[dict[str, Any], float]] = []
    deferred: list[tuple[dict[str, Any], float]] = []
    selected_ids: list[str] = []
    for item in ranked:
        skill_id = item[0]["skill_id"]
        if any(frozenset({skill_id, prior}) in similar_pairs for prior in selected_ids):
            deferred.append(item)
        else:
            selected.append(item)
            selected_ids.append(skill_id)
    return selected + deferred


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _task_metrics(
    ranking: list[tuple[dict[str, Any], float]],
    evidence: dict[str, dict[str, Any]],
    similar_pairs: set[frozenset[str]],
    ks: tuple[int, ...],
) -> dict[str, float]:
    judged = [
        (row, score, evidence[row["skill_id"]])
        for row, score in ranking
        if evidence[row["skill_id"]]["raw_label"] != "uncertain"
    ]
    ideal_grades = sorted(
        (GRADE[item["raw_label"]] for item in evidence.values() if item["raw_label"] != "uncertain"),
        reverse=True,
    )
    required_total = sum(item["raw_label"] == "required" for item in evidence.values())
    first_required_rank = next(
        (index for index, (_, _, item) in enumerate(judged, 1) if item["raw_label"] == "required"),
        None,
    )
    result = {"mrr_required": 1.0 / first_required_rank if first_required_rank else 0.0}
    for k in ks:
        top = judged[:k]
        grades = [GRADE[item["raw_label"]] for _, _, item in top]
        ideal = _dcg(ideal_grades[:k])
        required_found = sum(item["raw_label"] == "required" for _, _, item in top)
        bad = sum(item["raw_label"] in {"irrelevant", "harmful"} for _, _, item in top)
        harmful = sum(item["raw_label"] == "harmful" for _, _, item in top)
        top_ids = [row["skill_id"] for row, _, _ in top]
        redundant_items = sum(
            any(frozenset({skill_id, prior}) in similar_pairs for prior in top_ids[:index])
            for index, skill_id in enumerate(top_ids)
        )
        pair_count = len(top_ids) * (len(top_ids) - 1) // 2
        similar_count = sum(
            frozenset({top_ids[i], top_ids[j]}) in similar_pairs
            for i in range(len(top_ids))
            for j in range(i + 1, len(top_ids))
        )
        result.update(
            {
                f"ndcg@{k}": _dcg(grades) / ideal if ideal else 0.0,
                f"required_recall@{k}": required_found / required_total if required_total else 0.0,
                f"bad_item_rate@{k}": bad / len(top) if top else 0.0,
                f"harmful_item_rate@{k}": harmful / len(top) if top else 0.0,
                f"redundant_item_rate@{k}": redundant_items / len(top_ids) if top_ids else 0.0,
                f"similar_pair_rate@{k}": similar_count / pair_count if pair_count else 0.0,
            }
        )
    return result


def _clean_exclusions(
    tasks: dict[str, dict[str, Any]], evidence_by_task: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, list[str]]:
    exclusions: dict[str, list[str]] = defaultdict(list)
    redundant_words = re.compile(r"\b(redundant|alternative|equally valid|same .* capability)\b", re.I)
    for task_id, evidence in evidence_by_task.items():
        task = tasks[task_id]
        labels = [item["raw_label"] for item in evidence.values()]
        if labels and all(label in {"required", "helpful"} for label in labels):
            exclusions[task_id].append("all_candidates_positive")
        if any(
            item["raw_label"] == "required" and redundant_words.search(item.get("reason", ""))
            for item in evidence.values()
        ):
            exclusions[task_id].append("required_reason_indicates_redundancy")
        text = str(task.get("task_text", "")).lower()
        task_type = str(task.get("task_type", ""))
        if "pick_cool" in task_type and re.search(r"\b(heat|heated|cook|cooked)\b", text):
            exclusions[task_id].append("task_text_conflicts_with_cool_plan")
        if "pick_heat" in task_type and re.search(r"\b(cool|cold|chill|chilled)\b", text):
            exclusions[task_id].append("task_text_conflicts_with_heat_plan")
    return dict(exclusions)


def evaluate_reranking_baselines(
    tasks_path: Path | str,
    candidates_path: Path | str,
    evidence_path: Path | str,
    graph_path: Path | str,
    output_root: Path | str,
    *,
    ks: tuple[int, ...] = (3, 5, 10),
) -> dict[str, Any]:
    """Evaluate non-trained rerankers on full and automatically cleaned subsets."""
    tasks_path = Path(tasks_path).expanduser().resolve()
    candidates_path = Path(candidates_path).expanduser().resolve()
    evidence_path = Path(evidence_path).expanduser().resolve()
    graph_path = Path(graph_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    tasks = {row["record_id"]: row for row in _load_jsonl(tasks_path)}
    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(candidates_path):
        candidates_by_task[row["task_record_id"]].append(row)
    evidence_by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _load_jsonl(evidence_path):
        evidence_by_task[row["task_record_id"]][row["skill_id"]] = row
    if set(candidates_by_task) != set(evidence_by_task):
        raise ValueError("candidate and evidence task IDs do not match")
    for task_id, candidates in candidates_by_task.items():
        if {row["skill_id"] for row in candidates} != set(evidence_by_task[task_id]):
            raise ValueError(f"candidate and evidence skill IDs do not match for {task_id}")

    graph_raw = graph_path.read_bytes()
    graph = json.loads(graph_raw)
    similar_pairs = {
        frozenset({edge["source"], edge["target"]})
        for edge in graph.get("edges", [])
        if edge.get("type") == "similar_to"
    }
    exclusions = _clean_exclusions(tasks, evidence_by_task)
    subsets = {
        "all_weak": sorted(evidence_by_task),
        "clean": sorted(set(evidence_by_task) - set(exclusions)),
    }
    prediction_rows = []
    subset_reports: dict[str, Any] = {}
    for subset_name, task_ids in subsets.items():
        method_reports = {}
        for method in METHODS:
            per_task = []
            for task_id in task_ids:
                ranking = _rank(candidates_by_task[task_id], method, similar_pairs)
                per_task.append(
                    _task_metrics(ranking, evidence_by_task[task_id], similar_pairs, ks)
                )
                for rank, (row, score) in enumerate(ranking, 1):
                    evidence = evidence_by_task[task_id][row["skill_id"]]
                    prediction_rows.append(
                        {
                            "subset": subset_name,
                            "method": method,
                            "task_record_id": task_id,
                            "skill_id": row["skill_id"],
                            "rank": rank,
                            "ranking_score": score,
                            "judge_label": evidence["raw_label"],
                            "judge_grade": GRADE.get(evidence["raw_label"]),
                        }
                    )
            metric_names = sorted(per_task[0]) if per_task else []
            method_reports[method] = {
                name: sum(item[name] for item in per_task) / len(per_task)
                for name in metric_names
            }
        subset_reports[subset_name] = {
            "task_count": len(task_ids),
            "methods": method_reports,
        }

    benchmark_dir = output_root / "benchmarks" / "pilot_reranking"
    predictions_path = benchmark_dir / "rankings.jsonl"
    exclusions_path = benchmark_dir / "clean_exclusions.json"
    report_path = benchmark_dir / "report.json"
    _atomic_jsonl(predictions_path, prediction_rows)
    _atomic_json(exclusions_path, exclusions)
    report = {
        "schema_version": "skilldag_ncf.reranking_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_evidence_is_gold": False,
        "grade_mapping": GRADE,
        "uncertain_policy": "ignored",
        "methods": list(METHODS),
        "ks": list(ks),
        "similar_edge_count": len(similar_pairs),
        "clean_excluded_task_count": len(exclusions),
        "clean_exclusion_reason_counts": dict(
            sorted(Counter(reason for reasons in exclusions.values() for reason in reasons).items())
        ),
        "subsets": subset_reports,
        "provenance": {
            "tasks_path": str(tasks_path),
            "candidates_path": str(candidates_path),
            "evidence_path": str(evidence_path),
            "graph_path": str(graph_path),
            "graph_snapshot_id": f"sha256:{_sha256_bytes(graph_raw)}",
        },
        "outputs": {
            "rankings": str(predictions_path),
            "clean_exclusions": str(exclusions_path),
            "report": str(report_path),
        },
    }
    _atomic_json(report_path, report)
    return report
