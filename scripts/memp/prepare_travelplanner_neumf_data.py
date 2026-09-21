#!/usr/bin/env python3
"""Assemble train-only TravelPlanner Content-NeuMF arrays from V2 labels.

This is the TravelPlanner adapter for the same content-only 2/1/0 NeuMF
training contract used by the ALFWorld Memory--Task experiment.  It makes no
API calls and never reads validation/test tasks or trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / ".runtime/memp/travelplanner"
SPLIT_CODES = {"internal_train": 0, "internal_dev": 1}
SOURCE_NAMES = (
    "same_train_source",
    "query_cosine",
    "structure_match",
    "workflow_capability",
    "hard_negative",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def text_key(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_query_embeddings(path: Path, texts: list[str], model: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model:
        raise ValueError(f"query embedding model mismatch: {payload.get('model')!r}")
    records = payload.get("records") or {}
    vectors = []
    for text in texts:
        row = records.get(text_key(text))
        if row is None or row.get("text") != text:
            raise ValueError(f"missing query embedding for {text!r}")
        vectors.append(row["embedding"])
    return np.asarray(vectors, dtype=np.float32)


def load_memory_embeddings(path: Path, texts: list[str], model: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model:
        raise ValueError(f"memory embedding model mismatch: {payload.get('model')!r}")
    source_texts = payload.get("texts") or payload.get("queries") or []
    source_vectors = payload.get("vectors") or payload.get("embeddings") or []
    by_text: dict[str, list[float]] = {}
    for text, vector in zip(source_texts, source_vectors, strict=True):
        if text in by_text:
            raise ValueError(f"duplicate embedding text in {path}: {text!r}")
        by_text[text] = vector
    missing = [text for text in texts if text not in by_text]
    if missing:
        raise ValueError(f"missing {len(missing)} memory embeddings; first={missing[0]!r}")
    return np.asarray([by_text[text] for text in texts], dtype=np.float32)


def normalized(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-views",
        type=Path,
        default=RUNTIME / "ncf_v2/query_views/query_views.jsonl",
    )
    parser.add_argument(
        "--memory-profiles",
        type=Path,
        default=RUNTIME / "ncf_v2/memory_profiles/memory_capability_profiles.json",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=RUNTIME / "ncf_v2/candidates/judge_candidates.jsonl",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=RUNTIME / "ncf_v2/judge_full405_gpt4o_v3_calibrated_v2/labels.jsonl",
    )
    parser.add_argument(
        "--query-embedding-cache",
        type=Path,
        default=RUNTIME / "ncf_v2/candidates/embedding_cache/query_views.json",
    )
    parser.add_argument(
        "--memory-embedding-cache",
        type=Path,
        default=RUNTIME / "offline_v1_v5_validation180/v5/embeddings.json",
        help="Frozen source-query embeddings used by the paper Query-cosine baseline.",
    )
    parser.add_argument(
        "--memory-content-embedding-cache",
        type=Path,
        default=RUNTIME / "ncf_v2/memory_content_embeddings/text_embedding_3_small.json",
        help="Cache for '[memory query + workflow]' embeddings used only by Content-NeuMF.",
    )
    parser.add_argument(
        "--task-content-embedding-cache",
        type=Path,
        default=RUNTIME / "ncf_v2/capability_aligned_embeddings_v1/task_focus.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNTIME / "ncf_v2/model_data/content_neumf_query_only_v1",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--memory-representation",
        choices=("query_only", "query_plus_workflow", "capability_aligned"),
        default="query_only",
    )
    parser.add_argument("--train-negative-ratio", type=int, default=4)
    parser.add_argument("--expected-candidate-budget", type=int, default=20)
    parser.add_argument(
        "--forced-preferences",
        type=Path,
        help="Optional train-only anchor/rescue manifest whose pair endpoints must be retained.",
    )
    args = parser.parse_args()
    if args.train_negative_ratio <= 0:
        raise ValueError("train-negative-ratio must be positive")

    views = read_jsonl(args.query_views)
    profiles = json.loads(args.memory_profiles.read_text(encoding="utf-8"))
    batches = read_jsonl(args.candidates)
    labels = read_jsonl(args.labels)
    view_by_id = {index: row for index, row in enumerate(views)}
    profile_by_id = {int(row["memory_id"]): row for row in profiles}
    batch_by_task = {int(row["task_id"]): row for row in batches}
    labels_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        labels_by_task[int(row["task_id"])].append(row)
    expected_pairs = 405 * args.expected_candidate_budget
    if (
        len(views) != 405
        or len(profiles) != 45
        or len(batches) != 405
        or len(labels) != expected_pairs
    ):
        raise ValueError(
            f"expected frozen 405-query/45-memory/{expected_pairs}-pair inputs"
        )
    if set(view_by_id) != set(batch_by_task) or set(view_by_id) != set(labels_by_task):
        raise ValueError("query, candidate, and label task IDs differ")
    if set(profile_by_id) != set(range(45)):
        raise ValueError("memory IDs must be dense 0..44")
    if any(row.get("validation_or_test_used") is not False for row in views + profiles + batches + labels):
        raise ValueError("evaluation-derived row detected")

    candidate_by_pair = {
        (int(batch["task_id"]), int(candidate["memory_id"])): candidate
        for batch in batches
        for candidate in batch["candidates"]
    }
    forced_memory_ids_by_task: dict[int, set[int]] = defaultdict(set)
    forced_preference_rows: list[dict[str, Any]] = []
    if args.forced_preferences:
        forced_preference_rows = read_jsonl(args.forced_preferences)
        for row in forced_preference_rows:
            if row.get("source_split") != "internal_train":
                raise ValueError("forced preference is not internal_train")
            if row.get("validation_or_test_used") is not False:
                raise ValueError("evaluation-derived forced preference detected")
            task_id = int(row["task_id"])
            forced_memory_ids_by_task[task_id].update(
                (int(row["positive_memory_id"]), int(row["negative_memory_id"]))
            )
    usable_task_ids = sorted(
        task_id
        for task_id, rows in labels_by_task.items()
        if any(row["grade"] in {1, 2} for row in rows)
    )
    excluded_task_ids = sorted(set(view_by_id) - set(usable_task_ids))
    if len(usable_task_ids) + len(excluded_task_ids) != 405:
        raise ValueError("usable/excluded query partition is incomplete")

    # Family split is the unit of independence, not an individual derived view.
    family_splits: dict[str, set[str]] = defaultdict(set)
    for task_id in usable_task_ids:
        view = view_by_id[task_id]
        family_splits[view["family_id"]].add(view["model_split"])
    cross_split = {family: sorted(splits) for family, splits in family_splits.items() if len(splits) > 1}
    if cross_split:
        raise ValueError(f"families cross model splits: {cross_split}")

    selected: list[dict[str, Any]] = []
    available_pair_counts = []
    available_negative_counts = []
    selected_count_by_task: dict[int, int] = {}
    for task_id in usable_task_ids:
        rows = [row for row in labels_by_task[task_id] if row["grade"] != "uncertain"]
        positives = sorted(
            (row for row in rows if row["grade"] in {1, 2}),
            key=lambda row: (-int(row["grade"]), int(row["memory_id"])),
        )
        negatives = [row for row in rows if row["grade"] == 0]

        def negative_key(row: dict[str, Any]) -> tuple[Any, ...]:
            candidate = candidate_by_pair[(task_id, int(row["memory_id"]))]
            return (
                int(candidate.get("query_cosine_rank", 10**9)),
                -float(candidate.get("query_cosine_score", -1.0)),
                -float(candidate.get("structural_score", 0.0)),
                int(row["memory_id"]),
            )

        negatives.sort(key=negative_key)
        available_pair_counts.append(len(rows))
        available_negative_counts.append(len(negatives))
        if view_by_id[task_id]["model_split"] == "internal_dev":
            chosen = sorted(rows, key=lambda row: int(row["memory_id"]))
        else:
            chosen = positives + negatives[: args.train_negative_ratio * len(positives)]
            chosen_by_memory = {int(row["memory_id"]): row for row in chosen}
            available_by_memory = {int(row["memory_id"]): row for row in rows}
            for memory_id in sorted(forced_memory_ids_by_task.get(task_id, set())):
                if memory_id not in available_by_memory:
                    raise ValueError(
                        f"forced pair endpoint missing determinate label: task={task_id}, memory={memory_id}"
                    )
                chosen_by_memory[memory_id] = available_by_memory[memory_id]
            chosen = sorted(
                chosen_by_memory.values(),
                key=lambda row: (-int(row["grade"]), int(row["memory_id"])),
            )
        selected.extend(chosen)
        selected_count_by_task[task_id] = len(chosen)

    task_texts = [view_by_id[task_id]["query_text"] for task_id in usable_task_ids]
    memory_ids = sorted(profile_by_id)
    memory_query_texts = [profile_by_id[memory_id]["memory_query"] for memory_id in memory_ids]
    baseline_task_embeddings = load_query_embeddings(
        args.query_embedding_cache, task_texts, args.embedding_model
    )
    baseline_memory_embeddings = load_memory_embeddings(
        args.memory_embedding_cache, memory_query_texts, args.embedding_model
    )
    if args.memory_representation == "query_only":
        task_embeddings = baseline_task_embeddings.copy()
        memory_embeddings = baseline_memory_embeddings.copy()
        memory_model_input = "memory_source_query_embedding"
        model_memory_cache = args.memory_embedding_cache
        model_task_cache = args.query_embedding_cache
    elif args.memory_representation == "query_plus_workflow":
        task_embeddings = baseline_task_embeddings.copy()
        memory_content_texts = [
            f"Memory task: {profile_by_id[memory_id]['memory_query']}\n"
            f"Reusable procedure: {profile_by_id[memory_id]['workflow']}"
            for memory_id in memory_ids
        ]
        memory_embeddings = load_query_embeddings(
            args.memory_content_embedding_cache,
            memory_content_texts,
            args.embedding_model,
        )
        memory_model_input = "memory_source_query_plus_workflow_embedding"
        model_memory_cache = args.memory_content_embedding_cache
        model_task_cache = args.query_embedding_cache
    else:
        capability_dir = args.task_content_embedding_cache.parent
        task_payload = json.loads(args.task_content_embedding_cache.read_text(encoding="utf-8"))
        memory_capability_cache = capability_dir / "memory_capability.json"
        memory_payload = json.loads(memory_capability_cache.read_text(encoding="utf-8"))
        task_records = task_payload.get("records") or {}
        memory_records = memory_payload.get("records") or {}
        # The embedding script records texts in insertion order.  Reuse that
        # deterministic order while independently checking expected counts.
        task_embeddings = np.asarray(
            [row["embedding"] for row in task_records.values()], dtype=np.float32
        )
        memory_embeddings = np.asarray(
            [row["embedding"] for row in memory_records.values()], dtype=np.float32
        )
        if len(task_embeddings) != 405 or len(memory_embeddings) != 45:
            raise ValueError("capability-aligned embedding cache size mismatch")
        task_embeddings = task_embeddings[np.asarray(usable_task_ids, dtype=np.int32)]
        memory_model_input = "memory_capability_plus_workflow_embedding"
        model_memory_cache = memory_capability_cache
        model_task_cache = args.task_content_embedding_cache
    if task_embeddings.shape[1] != memory_embeddings.shape[1]:
        raise ValueError("task and memory embedding dimensions differ")
    # Keep the paper Query-cosine baseline independent of the NCF memory
    # representation.  Adding workflow content must never silently alter the
    # baseline score or its candidate ordering.
    cosine_matrix = normalized(baseline_task_embeddings) @ normalized(baseline_memory_embeddings).T

    task_index = {task_id: index for index, task_id in enumerate(usable_task_ids)}
    memory_index = {memory_id: index for index, memory_id in enumerate(memory_ids)}
    pair_count = len(selected)
    pair_task_indices = np.empty(pair_count, dtype=np.int32)
    pair_memory_indices = np.empty(pair_count, dtype=np.int16)
    target_grade = np.empty(pair_count, dtype=np.int8)
    target_relevance = np.empty(pair_count, dtype=np.float32)
    label_confidence = np.empty(pair_count, dtype=np.float32)
    split_codes = np.empty(pair_count, dtype=np.int8)
    source_features = np.empty((pair_count, len(SOURCE_NAMES)), dtype=np.float32)
    baseline_query_cosine = np.empty(pair_count, dtype=np.float32)
    pair_ids = []
    for offset, row in enumerate(selected):
        task_id = int(row["task_id"])
        memory_id = int(row["memory_id"])
        candidate = candidate_by_pair[(task_id, memory_id)]
        pair_task_indices[offset] = task_index[task_id]
        pair_memory_indices[offset] = memory_index[memory_id]
        target_grade[offset] = int(row["grade"])
        target_relevance[offset] = int(row["grade"]) / 2.0
        label_confidence[offset] = float(row["confidence"])
        split_codes[offset] = SPLIT_CODES[row["internal_split"]]
        sources = set(candidate["candidate_sources"])
        source_features[offset] = [float(name in sources) for name in SOURCE_NAMES]
        baseline_query_cosine[offset] = cosine_matrix[task_index[task_id], memory_index[memory_id]]
        pair_ids.append(f"{row['view_id']}::memory_{memory_id:03d}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "arrays.npz"
    temporary = args.output_dir / "arrays.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            task_ids=np.asarray(usable_task_ids, dtype=np.int32),
            memory_ids=np.asarray(memory_ids, dtype=np.int16),
            family_ids=np.asarray([view_by_id[x]["family_id"] for x in usable_task_ids]),
            view_types=np.asarray([view_by_id[x]["view_type"] for x in usable_task_ids]),
            available_pair_counts=np.asarray(available_pair_counts, dtype=np.int16),
            available_negative_counts=np.asarray(available_negative_counts, dtype=np.int16),
            selected_pair_counts=np.asarray([selected_count_by_task[x] for x in usable_task_ids], dtype=np.int16),
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
    temporary.replace(arrays_path)

    grade_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        grade_by_split[row["internal_split"]][str(row["grade"])] += 1
    task_split_counts = Counter(view_by_id[x]["model_split"] for x in usable_task_ids)
    metadata = {
        "schema_version": "memp.travelplanner.content_neumf_data.v1",
        "api_calls_made": 0,
        "uses_eval_data": False,
        "embedding_model": args.embedding_model,
        "embedding_dimension": int(task_embeddings.shape[1]),
        "memory_representation": args.memory_representation,
        "query_context_count_raw": len(views),
        "eligible_query_count": len(usable_task_ids),
        "excluded_query_count": len(excluded_task_ids),
        "excluded_query_ids": excluded_task_ids,
        "eligible_query_counts_by_split": dict(sorted(task_split_counts.items())),
        "memory_count": len(memory_ids),
        "pair_count": pair_count,
        "pair_counts_by_split": dict(Counter(row["internal_split"] for row in selected)),
        "grade_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(grade_by_split.items())
        },
        "split_codes": SPLIT_CODES,
        "candidate_source_feature_names": SOURCE_NAMES,
        "target_relevance": "target_grade / 2",
        "sample_weight_contract": "uniform_1.0; Judge confidence retained for audit only",
        "train_negative_sampling": {
            "negative_per_positive": args.train_negative_ratio,
            "policy": "all Grade-2/1 plus hardest Grade-0 by Query-cosine rank",
            "internal_dev_policy": "all determinate candidate labels",
        },
        "forced_preferences": {
            "path": str(args.forced_preferences.resolve()) if args.forced_preferences else None,
            "preference_count": len(forced_preference_rows),
            "forced_endpoint_count": sum(len(x) for x in forced_memory_ids_by_task.values()),
            "policy": "retain both train-only anchor/rescue endpoints; no label or weight changes",
        },
        "content_only_contract": {
            "learned_task_id_embedding": False,
            "learned_memory_id_embedding": False,
            "model_inputs": ["focused_task_query_embedding", memory_model_input],
            "baseline_cosine_inputs": [
                "focused_task_query_embedding",
                "memory_source_query_embedding",
            ],
            "baseline_cosine_is_unchanged_by_memory_representation": True,
            "memory_workflow_used_by_neumf": args.memory_representation == "query_plus_workflow",
            "ids_used_only_as_array_indices": True,
        },
        "privileged_fields_excluded_from_model_inputs": [
            "judge_reason",
            "focus_evidence",
            "label_confidence",
            "validation_or_test_task",
            "validation_or_test_trajectory",
            "reward",
        ],
        "integrity": {
            "family_cross_split_count": 0,
            "query_views_sha256": sha256_file(args.query_views),
            "memory_profiles_sha256": sha256_file(args.memory_profiles),
            "candidates_sha256": sha256_file(args.candidates),
            "calibrated_labels_sha256": sha256_file(args.labels),
            "query_embeddings_sha256": sha256_file(args.query_embedding_cache),
            "memory_embeddings_sha256": sha256_file(args.memory_embedding_cache),
            "model_memory_embeddings_sha256": sha256_file(model_memory_cache),
            "model_task_embeddings_sha256": sha256_file(model_task_cache),
        },
        "outputs": {"arrays": "arrays.npz", "metadata": "metadata.json"},
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
