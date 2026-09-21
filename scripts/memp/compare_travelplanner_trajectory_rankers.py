#!/usr/bin/env python3
"""Offline, zero-API sanity comparison of trajectory-document rankers.

For a validation slice (default 10-19, the pilot slice), rank the 23
{trajectory,proceduralization} documents for every task query with three
scorers and compare against the deterministic signature-similarity grades
used as route-B training labels:

* query cosine      - frozen baseline embedding score
* plugged V25f      - script-trained NeuMF applied to whole documents (OOD)
* retrained format  - route-B NeuMF trained on this format's 23 documents

Metrics are NDCG@k and top-3 grade hits against the deterministic labels
(not runtime gold).  This script makes no API call (cached embeddings only).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
NCF_SRC = ROOT / "src" / "GraphOfSkills_NCF"
if str(NCF_SRC) not in sys.path:
    sys.path.insert(0, str(NCF_SRC))

from build_memory_embedding_candidates import embed_texts  # noqa: E402
from prepare_travelplanner_phase_dynamic_retrieval import (  # noqa: E402
    score_neumf_queries,
    task_signature,
)
from gos.ncf.models import NCFConfig, NeuMF  # noqa: E402

CONSTRAINT_FIELDS = ("transportation", "cuisine", "room_type", "house_rule")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_signature(row: dict[str, str]) -> dict[str, Any]:
    constraints = ast.literal_eval(row["local_constraint"])
    days, people = int(row["days"]), int(row["people_number"])
    return {
        "people_number": people,
        "budget_per_person_day": int(row["budget"]) / (people * days),
        "transportation": constraints.get("transportation"),
        "cuisine": constraints.get("cuisine"),
        "room_type": constraints.get("room type"),
        "house_rule": constraints.get("house rule"),
    }


def grade_for(query_signature: dict[str, Any], source: dict[str, Any]) -> int:
    match = mismatch = 0
    people = query_signature.get("people_number")
    if people == source["people_number"]:
        match += 1
    elif people is not None:
        mismatch += 1
    budget_per_day = source["budget_per_person_day"]
    query_budget = query_signature.get("budget_per_person_day")
    if query_budget is not None:
        if abs(query_budget - budget_per_day) < 0.5 * max(budget_per_day, 1.0):
            match += 1
        else:
            mismatch += 1
    for field in CONSTRAINT_FIELDS:
        query_value = query_signature.get(field)
        source_value = source.get(field)
        if query_value and source_value and query_value == source_value:
            match += 1
        elif source_value and not query_value:
            mismatch += 1
    if mismatch == 0 and match >= 2:
        return 2
    if mismatch == 0 or match >= 1:
        return 1
    return 0


def ndcg_at_k(scores: np.ndarray, grades: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    top_grades = grades[order]
    gains = top_grades.astype(np.float64)
    dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = np.sort(grades)[::-1][:k].astype(np.float64)
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def top_k_metrics(scores: np.ndarray, grades: np.ndarray, k: int) -> dict[str, float]:
    order = np.argsort(-scores)[:k]
    top = grades[order]
    return {
        f"hit2@{k}": float(np.mean(top == 2)),
        f"bad0@{k}": float(np.mean(top == 0)),
        f"avg_grade@{k}": float(np.mean(top)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-texts", type=Path, required=True)
    parser.add_argument("--format", choices=("trajectory", "proceduralization"), required=True)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--task-start", type=int, default=10)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plugged-checkpoint", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    args = parser.parse_args()

    content_rows = read_jsonl(args.content_texts)
    documents = [row for row in content_rows if row["format"] == args.format]
    if len(documents) != 23:
        raise ValueError("expected 23 documents")
    train_rows = list(csv.DictReader(args.train_csv.open(encoding="utf-8", newline="")))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(manifest["items"]) != 23:
        raise ValueError("expected 23 manifest items")
    source_by_memory = [
        source_signature(train_rows[int(str(row["source"]).rsplit("_", 1)[-1])])
        for row in documents
    ]

    table = list(csv.DictReader(args.task_csv.open(encoding="utf-8", newline="")))
    tasks = [
        (index, str(table[index]["query"]), task_signature(table[index]))
        for index in range(args.task_start, min(args.task_start + args.task_count, len(table)))
    ]

    texts = [str(row["content_text"])[:8000] for row in documents] + [query for _, query, _ in tasks]
    vectors = embed_texts(
        texts,
        model="text-embedding-3-small",
        cache_path=args.cache_dir / "trajectory_content_embeddings.json",
        api_base=os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1"),
        api_key=os.environ.get("SKILLDAG_EMBEDDING_API_KEY", ""),
        batch_size=64,
        timeout=180,
    ).astype(np.float32)
    doc_vectors = vectors[:23]
    query_vectors = vectors[23:]
    cosine = query_vectors @ doc_vectors.T

    retrained_model = load_neumf(args.checkpoint)
    plugged_model = load_neumf(args.plugged_checkpoint)
    with torch.no_grad():
        retrained_scores = score_neumf_queries(retrained_model, query_vectors, doc_vectors, 16)
        plugged_scores = score_neumf_queries(plugged_model, query_vectors, doc_vectors, 16)

    rows_out: list[dict[str, Any]] = []
    aggregate: dict[str, list[float]] = defaultdict(list)
    for query_index, (task_index, task, query_signature) in enumerate(tasks):
        grades = np.asarray(
            [grade_for(query_signature, source_by_memory[i]) for i in range(23)], dtype=np.int64
        )
        for name, scores in (
            ("query_cosine", cosine[query_index]),
            ("plugged_v25f", plugged_scores[query_index]),
            ("retrained_format", retrained_scores[query_index]),
        ):
            values = top_k_metrics(scores, grades, 3)
            values["ndcg@3"] = ndcg_at_k(scores, grades, 3)
            values["ndcg@5"] = ndcg_at_k(scores, grades, 5)
            values["ndcg@10"] = ndcg_at_k(scores, grades, 10)
            for key, value in values.items():
                aggregate[f"{name}:{key}"].append(value)
            rows_out.append(
                {
                    "task_index": task_index,
                    "query": task,
                    "ranker": name,
                    "top3_ids": [documents[i]["memory_id"] for i in np.argsort(-scores)[:3]],
                    "top3_grades": [int(g) for g in grades[np.argsort(-scores)[:3]]],
                    **{key: float(value) for key, value in values.items()},
                }
            )
    summary = {key: float(np.mean(values)) for key, values in sorted(aggregate.items())}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows_out),
            encoding="utf-8",
        )
    return 0


def load_neumf(path: Path) -> NeuMF:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = NeuMF(NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    raise SystemExit(main())
