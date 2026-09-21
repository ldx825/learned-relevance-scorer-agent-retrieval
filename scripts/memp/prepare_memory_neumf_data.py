#!/usr/bin/env python3
"""Assemble leakage-safe Content-NeuMF arrays from calibrated Memory labels.

The script makes no API calls.  It consumes frozen text-embedding-3-large
caches and excludes task groups for which the fixed memory bank has no
positive (Grade 1 or Grade 2) supervision.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_CODES = {"train": 0, "model_dev": 1}
SOURCE_NAMES = (
    "query_cosine",
    "resource_content_cosine",
    "structured_exact",
    "structured_complement",
    "structured_hard_negative",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def text_key(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embedding_cache(path: Path, texts: list[str], model: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model:
        raise ValueError(f"embedding model mismatch in {path}: {payload.get('model')!r}")
    records = payload.get("records") or {}
    values = []
    for text in texts:
        key = text_key(text)
        row = records.get(key)
        if not row or row.get("text") != text:
            raise ValueError(f"missing or mismatched embedding for {text!r} in {path}")
        values.append(row["embedding"])
    return np.asarray(values, dtype=np.float32)


def load_baseline_memory_embeddings(
    path: Path, memory_queries: list[str], model: str
) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model:
        raise ValueError(f"baseline embedding model mismatch in {path}")
    if payload.get("queries") != memory_queries:
        raise ValueError("baseline memory query order/text differs from frozen bank")
    values = np.asarray(payload.get("embeddings"), dtype=np.float32)
    if len(values) != len(memory_queries):
        raise ValueError("baseline memory embedding count mismatch")
    return values


def normalized(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--query-views",
        type=Path,
        help="Optional query_views.jsonl; when set, each focused view is a task-side training query.",
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--calibrated-pairs", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--task-embedding-cache",
        type=Path,
        help="Optional explicit cache containing task/query-view embeddings.",
    )
    parser.add_argument("--baseline-memory-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--baseline-embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--memory-representation",
        choices=("query_only", "query_plus_workflow"),
        default="query_plus_workflow",
        help="Text representation used by both NeuMF towers.",
    )
    parser.add_argument("--train-negative-ratio", type=int, default=4)
    args = parser.parse_args()

    if args.query_views:
        tasks = [
            {
                "task_id": int(row["query_view_index"]),
                "query": row["query_text"],
                "model_split": row["model_split"],
                "canonical_id": row["canonical_id"],
                "parent_task_id": int(row["parent_task_id"]),
                "view_type": row["view_type"],
            }
            for row in load_jsonl(args.query_views)
        ]
    else:
        tasks = load_jsonl(args.dataset / "tasks.jsonl")
    memories = load_jsonl(args.dataset / "memories.jsonl")
    batches = load_jsonl(args.candidates)
    pairs = load_jsonl(args.calibrated_pairs)
    task_by_id = {int(row["task_id"]): row for row in tasks}
    memory_by_id = {int(row["memory_id"]): row for row in memories}
    batch_by_task = {int(row["task_id"]): row for row in batches}
    if len(task_by_id) != len(tasks) or len(memory_by_id) != len(memories):
        raise ValueError("duplicate task_id or memory_id")
    if set(task_by_id) != set(batch_by_task):
        raise ValueError("candidate tasks do not exactly match the train-only dataset")

    pairs_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        task_id = int(row["task_id"])
        memory_id = int(row["memory_id"])
        if task_id not in task_by_id or memory_id not in memory_by_id:
            raise ValueError(f"unknown pair key {(task_id, memory_id)}")
        pairs_by_task[task_id].append(row)

    candidate_by_key = {
        (int(batch["task_id"]), int(candidate["memory_id"])): candidate
        for batch in batches
        for candidate in batch["candidates"]
    }

    eligible_task_ids = sorted(
        task_id
        for task_id, rows in pairs_by_task.items()
        if any(int(row["target_grade"]) > 0 for row in rows)
    )
    excluded_task_ids = sorted(set(task_by_id) - set(eligible_task_ids))
    selected_pairs: list[dict[str, Any]] = []
    for task_id in eligible_task_ids:
        rows = pairs_by_task[task_id]
        if task_by_id[task_id]["model_split"] == "model_dev":
            selected_pairs.extend(sorted(rows, key=lambda row: int(row["memory_id"])))
            continue
        positives = sorted(
            (row for row in rows if int(row["target_grade"]) > 0),
            key=lambda row: (-int(row["target_grade"]), int(row["memory_id"])),
        )
        negatives = [row for row in rows if int(row["target_grade"]) == 0]

        def negative_key(row: dict[str, Any]) -> tuple[float, ...]:
            candidate = candidate_by_key[(task_id, int(row["memory_id"]))]
            sources = set(candidate["candidate_sources"])
            if "query_cosine_fill" in sources:
                sources.add("query_cosine")
            ranks = [
                int(candidate[key])
                for key in ("query_cosine_rank", "content_cosine_rank")
                if key in candidate
            ]
            scores = [
                float(candidate[key])
                for key in ("query_cosine_score", "content_cosine_score")
                if key in candidate
            ]
            return (
                -float("query_cosine" in sources or "resource_content_cosine" in sources),
                float(min(ranks) if ranks else 10**6),
                -float(max(scores) if scores else -1.0),
                float(row["memory_id"]),
            )

        negatives.sort(key=negative_key)
        negative_limit = args.train_negative_ratio * len(positives)
        selected_pairs.extend(positives + negatives[:negative_limit])

    # Verify family-safe splits directly on canonical signatures.
    signature_splits: dict[str, set[str]] = defaultdict(set)
    for task_id in eligible_task_ids:
        task = task_by_id[task_id]
        signature_splits[task["canonical_id"]].add(task["model_split"])
    cross_split = {key: sorted(value) for key, value in signature_splits.items() if len(value) > 1}
    if cross_split:
        raise ValueError(f"canonical task signatures cross splits: {cross_split}")

    task_texts = [task_by_id[task_id]["query"] for task_id in eligible_task_ids]
    memory_ids = sorted(memory_by_id)
    memory_texts = [
        f"Memory task: {memory_by_id[memory_id]['query']}\n"
        f"Reusable procedure: {memory_by_id[memory_id]['workflow']}"
        for memory_id in memory_ids
    ]
    task_embedding_cache = args.task_embedding_cache or (
        args.embedding_cache_dir / "task_query_small.json"
    )
    task_query_embeddings = load_embedding_cache(
        task_embedding_cache,
        task_texts,
        args.baseline_embedding_model,
    )
    memory_query_embeddings = load_baseline_memory_embeddings(
        args.baseline_memory_embeddings,
        [memory_by_id[memory_id]["query"] for memory_id in memory_ids],
        args.baseline_embedding_model,
    )
    if args.memory_representation == "query_only":
        # Paper-aligned representation: compare the current task query with
        # the source task query attached to each memory.  The workflow remains
        # available to the Agent after retrieval, but is not an NCF input.
        task_embeddings = task_query_embeddings.copy()
        memory_embeddings = memory_query_embeddings.copy()
        model_embedding_name = args.baseline_embedding_model
        model_input_texts = ["task_query", "memory_source_query"]
    else:
        task_embeddings = load_embedding_cache(
            args.embedding_cache_dir / "task_query_large.json",
            task_texts,
            args.embedding_model,
        )
        memory_embeddings = load_embedding_cache(
            args.embedding_cache_dir / "memory_content_large.json",
            memory_texts,
            args.embedding_model,
        )
        model_embedding_name = args.embedding_model
        model_input_texts = ["task_query", "memory_query_plus_workflow"]
    if task_embeddings.shape[1] != memory_embeddings.shape[1]:
        raise ValueError("task and memory embedding dimensions differ")
    baseline_score_matrix = normalized(task_query_embeddings) @ normalized(
        memory_query_embeddings
    ).T

    task_index = {task_id: index for index, task_id in enumerate(eligible_task_ids)}
    memory_index = {memory_id: index for index, memory_id in enumerate(memory_ids)}
    available_pair_counts = np.asarray(
        [len(pairs_by_task[task_id]) for task_id in eligible_task_ids], dtype=np.int16
    )
    available_negative_counts = np.asarray(
        [
            sum(int(row["target_grade"]) == 0 for row in pairs_by_task[task_id])
            for task_id in eligible_task_ids
        ],
        dtype=np.int16,
    )
    pair_task_indices = np.empty(len(selected_pairs), dtype=np.int32)
    pair_memory_indices = np.empty(len(selected_pairs), dtype=np.int16)
    target_grade = np.empty(len(selected_pairs), dtype=np.int8)
    target_relevance = np.empty(len(selected_pairs), dtype=np.float32)
    label_confidence = np.empty(len(selected_pairs), dtype=np.float32)
    split_codes = np.empty(len(selected_pairs), dtype=np.int8)
    source_features = np.empty((len(selected_pairs), len(SOURCE_NAMES)), dtype=np.float32)
    baseline_query_cosine = np.empty(len(selected_pairs), dtype=np.float32)
    pair_ids: list[str] = []
    for offset, row in enumerate(selected_pairs):
        task_id = int(row["task_id"])
        memory_id = int(row["memory_id"])
        candidate = candidate_by_key[(task_id, memory_id)]
        pair_task_indices[offset] = task_index[task_id]
        pair_memory_indices[offset] = memory_index[memory_id]
        target_grade[offset] = int(row["target_grade"])
        target_relevance[offset] = float(row["target_grade"]) / 2.0
        label_confidence[offset] = float(row["label_confidence"])
        split_codes[offset] = SPLIT_CODES[row["internal_split"]]
        sources = set(candidate["candidate_sources"])
        if "query_cosine_fill" in sources:
            sources.add("query_cosine")
        if "structured_primary" in sources:
            sources.add("structured_exact")
        if "structured_support" in sources:
            sources.add("structured_complement")
        source_features[offset] = [float(name in sources) for name in SOURCE_NAMES]
        baseline_query_cosine[offset] = baseline_score_matrix[
            task_index[task_id], memory_index[memory_id]
        ]
        pair_ids.append(row["pair_id"])

    grade_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected_pairs:
        grade_by_split[row["internal_split"]][str(row["target_grade"])] += 1
    task_split_counts = Counter(task_by_id[task_id]["model_split"] for task_id in eligible_task_ids)
    excluded_split_counts = Counter(task_by_id[task_id]["model_split"] for task_id in excluded_task_ids)
    all_train_query_count = len(tasks)
    memory_count = len(memory_ids)
    pair_count = len(selected_pairs)
    pair_split_counts = Counter(row["internal_split"] for row in selected_pairs)

    # Large augmented datasets otherwise retain the raw JSON rows, multiple
    # lookup maps and the final NumPy arrays at the same time.  Everything
    # below this point is expressed by compact arrays and aggregate counters.
    del pairs, pairs_by_task, batches, batch_by_task, candidate_by_key
    del selected_pairs, task_by_id, memory_by_id, tasks, memories
    gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "arrays.npz"
    arrays_temp = args.output_dir / "arrays.npz.tmp"
    save_arrays = np.savez if len(eligible_task_ids) >= 10_000 else np.savez_compressed
    with arrays_temp.open("wb") as handle:
        save_arrays(
        handle,
        task_ids=np.asarray(eligible_task_ids, dtype=np.int32),
        memory_ids=np.asarray(memory_ids, dtype=np.int16),
        available_pair_counts=available_pair_counts,
        available_negative_counts=available_negative_counts,
        pair_ids=np.asarray(pair_ids),
        task_embeddings=task_embeddings,
        memory_embeddings=memory_embeddings,
        pair_task_indices=pair_task_indices,
        pair_memory_indices=pair_memory_indices,
        target_grade=target_grade,
        target_relevance=target_relevance,
        label_confidence=label_confidence,
        sample_weight=np.ones(pair_count, dtype=np.float32),
        split_codes=split_codes,
        candidate_source_features=source_features,
        baseline_query_cosine=baseline_query_cosine,
        )
    arrays_temp.replace(arrays_path)
    metadata = {
        "schema_version": "task_resource_ncf.memory_content_neumf_data.v1",
        "api_calls_made": 0,
        "uses_eval_data": False,
        "embedding_model": model_embedding_name,
        "memory_representation": args.memory_representation,
        "query_view_mode": bool(args.query_views),
        "baseline_embedding_model": args.baseline_embedding_model,
        "embedding_dimension": int(task_embeddings.shape[1]),
        "all_train_query_count": all_train_query_count,
        "eligible_query_count": len(eligible_task_ids),
        "excluded_all_zero_query_count": len(excluded_task_ids),
        "eligible_query_counts_by_split": dict(sorted(task_split_counts.items())),
        "excluded_query_counts_by_split": dict(sorted(excluded_split_counts.items())),
        "memory_count": memory_count,
        "pair_count": pair_count,
        "pair_counts_by_split": dict(sorted(pair_split_counts.items())),
        "grade_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(grade_by_split.items())
        },
        "split_codes": SPLIT_CODES,
        "candidate_source_feature_names": SOURCE_NAMES,
        "target_relevance": "target_grade / 2",
        "sample_weight_contract": "uniform_1.0; Judge confidence is retained for audit only",
        "train_negative_sampling": {
            "negative_per_positive": args.train_negative_ratio,
            "policy": "retain every Grade 2/1 and deterministically select the hardest Grade 0 candidates by cosine-route rank",
            "model_dev_policy": "retain the full candidate set",
            "alignment": "standard NCF implicit-feedback negative sampling; no source-quality weights",
        },
        "eligibility_contract": "retain a query group iff the fixed bank has at least one calibrated Grade 1 or Grade 2 candidate",
        "content_only_contract": {
            "learned_task_id_embedding": False,
            "learned_memory_id_embedding": False,
            "model_inputs": [f"{name}_embedding" for name in model_input_texts],
            "residual_base_score": "paper Query cosine: task query x memory source query",
            "ids_used_only_as_array_indices": True,
        },
        "privileged_fields_excluded_from_model_inputs": [
            "judge_reason",
            "raw_judge_grade",
            "calibration_action",
            "label_confidence",
            "reward",
            "dev_or_test_trajectory",
        ],
        "integrity": {
            "canonical_cross_split_count": 0,
            "candidate_dataset_sha256": sha256_file(args.candidates),
            "calibrated_pairs_sha256": sha256_file(args.calibrated_pairs),
            "task_embedding_cache_sha256": sha256_file(
                task_embedding_cache
                if args.memory_representation == "query_only"
                else args.embedding_cache_dir / "task_query_large.json"
            ),
            "memory_embedding_cache_sha256": sha256_file(
                args.baseline_memory_embeddings
                if args.memory_representation == "query_only"
                else args.embedding_cache_dir / "memory_content_large.json"
            ),
            "task_query_baseline_cache_sha256": sha256_file(task_embedding_cache),
            "memory_query_baseline_cache_sha256": sha256_file(args.baseline_memory_embeddings),
        },
        "outputs": {"arrays": "arrays.npz", "metadata": "metadata.json"},
        "array_storage": (
            "uncompressed_npz" if save_arrays is np.savez else "compressed_npz"
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
