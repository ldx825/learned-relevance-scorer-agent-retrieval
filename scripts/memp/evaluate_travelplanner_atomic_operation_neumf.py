#!/usr/bin/env python3
"""Paired internal-dev audit for atomic-operation cosine versus NeuMF."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
NCF_SRC = ROOT / "src/GraphOfSkills_NCF"
if str(NCF_SRC) not in sys.path:
    sys.path.insert(0, str(NCF_SRC))
from gos.ncf.models import NCFConfig, NeuMF  # noqa: E402


def dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def summarize(groups: dict[int, list[tuple[float, str, int]]]) -> tuple[dict, dict[int, dict]]:
    rows = []
    details = {}
    for task_index, candidates in groups.items():
        ranked = sorted(candidates, key=lambda row: (-row[0], row[1]))
        grades = [row[2] for row in ranked]
        ideal = sorted(grades, reverse=True)
        first_required = next((rank for rank, grade in enumerate(grades, 1) if grade == 2), None)
        detail = {
            "ret@1": float(any(grade == 2 for grade in grades[:1])),
            "ret@3": float(any(grade == 2 for grade in grades[:3])),
            "mrr": 1.0 / first_required if first_required else 0.0,
            "ndcg@3": dcg(grades[:3]) / max(dcg(ideal[:3]), 1e-12),
            "precision@3": sum(grade > 0 for grade in grades[:3]) / min(3, len(grades)),
            "bad@3": sum(grade == 0 for grade in grades[:3]) / min(3, len(grades)),
            "required_rank": first_required,
            "top_operation_id": ranked[0][1],
            "top_grade": ranked[0][2],
        }
        details[task_index] = detail
        rows.append(detail)
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("ret@1", "ret@3", "mrr", "ndcg@3", "precision@3", "bad@3")
    }
    metrics["query_count"] = len(rows)
    return metrics, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.arrays, allow_pickle=False) as values:
        data = {name: values[name].copy() for name in values.files}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NeuMF(NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    indices = np.flatnonzero(data["split_codes"] == 1)
    task_indices = data["pair_task_indices"][indices]
    operation_indices = data["pair_memory_indices"][indices]
    task = data["task_embeddings"][task_indices].astype(np.float32)
    operation = data["memory_embeddings"][operation_indices].astype(np.float32)
    cosine = np.sum(task * operation, axis=1) / np.maximum(
        np.linalg.norm(task, axis=1) * np.linalg.norm(operation, axis=1), 1e-12
    )
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            chunks.append(model(torch.from_numpy(task[start:start + 512]), torch.from_numpy(operation[start:start + 512])).numpy())
    neumf = np.concatenate(chunks)

    operation_ids = data["operation_ids"]
    cosine_groups: defaultdict[int, list[tuple[float, str, int]]] = defaultdict(list)
    neumf_groups: defaultdict[int, list[tuple[float, str, int]]] = defaultdict(list)
    for offset, pair_index in enumerate(indices):
        task_index = int(data["pair_task_indices"][pair_index])
        operation_index = int(data["pair_memory_indices"][pair_index])
        operation_id = str(operation_ids[operation_index])
        grade = int(data["target_grade"][pair_index])
        cosine_groups[task_index].append((float(cosine[offset]), operation_id, grade))
        neumf_groups[task_index].append((float(neumf[offset]), operation_id, grade))

    cosine_metrics, cosine_details = summarize(cosine_groups)
    neumf_metrics, neumf_details = summarize(neumf_groups)
    flips = Counter()
    for task_index in sorted(cosine_details):
        left = cosine_details[task_index]["ret@1"] == 1
        right = neumf_details[task_index]["ret@1"] == 1
        flips["both_success" if left and right else "cosine_only" if left else "neumf_only" if right else "both_fail"] += 1
    delta = {
        key: neumf_metrics[key] - cosine_metrics[key]
        for key in ("ret@1", "ret@3", "mrr", "ndcg@3", "precision@3", "bad@3")
    }
    report = {
        "schema_version": "memp.travelplanner.atomic_operation_paired_eval.v1",
        "scope": "train-derived internal-family-dev compact candidate groups",
        "cosine": cosine_metrics,
        "content_neumf": neumf_metrics,
        "delta_neumf_minus_cosine": delta,
        "ret1_paired_flips": dict(flips),
        "candidate_group_size_counts": dict(Counter(str(len(rows)) for rows in cosine_groups.values())),
        "caveat": "This diagnoses operation ranking only; it is not the official TravelPlanner #CS/#HC evaluation.",
        "validation_or_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
