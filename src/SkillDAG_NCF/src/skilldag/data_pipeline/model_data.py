"""Leakage-aware, NumPy-based model inputs for task-skill reranking."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .action_candidates import _atomic_json, _load_jsonl, _sha256_bytes


EDGE_TYPES = ("similar_to", "depends_on", "composes_with", "specializes")
DATASET_SPLITS = {"train": 0, "dev": 1, "test": 2}
PRIVILEGED_FIELDS = (
    "action_cosine_score",
    "action_semantic_rank",
    "matched_actions",
    "judge_confidence",
    "judge_reason",
    "expert_plan_visible",
    "source_model",
    "source_prompt_version",
)


def _load_task_embedding_cache(cache_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(cache_dir.glob("shard_*.json")):
        shard = json.loads(path.read_text(encoding="utf-8"))
        for task_id, entry in shard.items():
            if task_id in result and result[task_id] != entry:
                raise ValueError(f"conflicting cached task embedding: {task_id}")
            result[task_id] = entry
    return result


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding matrix contains a zero vector")
    return matrix / norms


def prepare_model_data(
    dataset_dir: Path | str,
    task_embedding_cache_dir: Path | str,
    skill_embedding_cache_path: Path | str,
    graph_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Align local embeddings and build student-only numeric features.

    No embedder or network callable is accepted by this function. All vectors
    must already exist in the local caches.
    """
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    task_embedding_cache_dir = Path(task_embedding_cache_dir).expanduser().resolve()
    skill_embedding_cache_path = Path(skill_embedding_cache_path).expanduser().resolve()
    graph_path = Path(graph_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    task_rows = _load_jsonl(dataset_dir / "tasks.jsonl")
    skill_rows = _load_jsonl(dataset_dir / "skills.jsonl")
    pair_rows = _load_jsonl(dataset_dir / "pairs.jsonl")
    tasks = {row["record_id"]: row for row in task_rows}
    skills = {row["skill_id"]: row for row in skill_rows}
    if len(tasks) != len(task_rows) or len(skills) != len(skill_rows):
        raise ValueError("duplicate task or skill IDs in normalized dataset")

    task_cache = _load_task_embedding_cache(task_embedding_cache_dir)
    skill_cache = json.loads(skill_embedding_cache_path.read_text(encoding="utf-8"))
    missing_tasks = set(tasks) - set(task_cache)
    missing_skills = set(skills) - set(skill_cache)
    if missing_tasks or missing_skills:
        raise ValueError(
            f"embedding cache incomplete: missing tasks={len(missing_tasks)}, skills={len(missing_skills)}"
        )
    models = {
        entry.get("model", "") for entry in task_cache.values() if entry.get("embedding")
    } | {entry.get("model", "") for entry in skill_cache.values() if entry.get("embedding")}
    if len(models) != 1:
        raise ValueError(f"task/skill embedding model mismatch: {sorted(models)}")

    task_ids = sorted(tasks)
    skill_ids = sorted(skills)
    task_index = {value: index for index, value in enumerate(task_ids)}
    skill_index = {value: index for index, value in enumerate(skill_ids)}
    task_embeddings = np.asarray(
        [task_cache[value]["embedding"] for value in task_ids], dtype=np.float32
    )
    skill_embeddings = np.asarray(
        [skill_cache[value]["embedding"] for value in skill_ids], dtype=np.float32
    )
    if task_embeddings.ndim != 2 or skill_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional matrices")
    if task_embeddings.shape[1] != skill_embeddings.shape[1]:
        raise ValueError("task and skill embedding dimensions differ")
    normalized_tasks = _normalize_rows(task_embeddings)
    normalized_skills = _normalize_rows(skill_embeddings)

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    degree: dict[str, Counter[str]] = defaultdict(Counter)
    for edge in graph.get("edges", []):
        source, target, edge_type = edge.get("source"), edge.get("target"), edge.get("type")
        if edge_type not in EDGE_TYPES:
            continue
        degree[str(source)][f"out_{edge_type}"] += 1
        degree[str(target)][f"in_{edge_type}"] += 1
    feature_names = ["cosine", "cosine_reciprocal_rank", "cosine_rank_fraction"]
    feature_names.extend(
        f"graph_{direction}_{edge_type}_degree_fraction"
        for edge_type in EDGE_TYPES
        for direction in ("out", "in")
    )
    feature_names.append("graph_total_degree_fraction")
    max_degree = max(len(skill_ids) - 1, 1)

    cosine_matrix = normalized_tasks @ normalized_skills.T
    ranks = np.empty_like(cosine_matrix, dtype=np.int16)
    for task_offset, scores in enumerate(cosine_matrix):
        order = np.lexsort((np.asarray(skill_ids), -scores))
        ranks[task_offset, order] = np.arange(1, len(skill_ids) + 1, dtype=np.int16)

    pair_task_indices = np.empty(len(pair_rows), dtype=np.int32)
    pair_skill_indices = np.empty(len(pair_rows), dtype=np.int16)
    pair_features = np.empty((len(pair_rows), len(feature_names)), dtype=np.float32)
    target_grade = np.empty(len(pair_rows), dtype=np.int8)
    target_relevance = np.empty(len(pair_rows), dtype=np.float32)
    sample_weight = np.empty(len(pair_rows), dtype=np.float32)
    split_codes = np.empty(len(pair_rows), dtype=np.int8)
    pair_ids: list[str] = []
    label_counts: Counter[str] = Counter()
    for offset, row in enumerate(pair_rows):
        ti = task_index[row["task_record_id"]]
        si = skill_index[row["skill_id"]]
        rank = int(ranks[ti, si])
        values = [
            float(cosine_matrix[ti, si]),
            1.0 / rank,
            (rank - 1) / max_degree,
        ]
        skill_degree = degree[row["skill_id"]]
        for edge_type in EDGE_TYPES:
            for direction in ("out", "in"):
                values.append(skill_degree[f"{direction}_{edge_type}"] / max_degree)
        values.append(sum(skill_degree.values()) / max_degree)
        pair_task_indices[offset] = ti
        pair_skill_indices[offset] = si
        pair_features[offset] = values
        target_grade[offset] = -1 if row["target_grade"] is None else int(row["target_grade"])
        target_relevance[offset] = float(row["target_relevance"])
        sample_weight[offset] = float(row["sample_weight"])
        split_codes[offset] = DATASET_SPLITS[row["dataset_split"]]
        pair_ids.append(row["pair_id"])
        label_counts[row["raw_label"]] += 1

    model_dir = output_root / "model_data" / "task_skill_v1"
    model_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = model_dir / "arrays.npz"
    metadata_path = model_dir / "metadata.json"
    np.savez_compressed(
        arrays_path,
        task_ids=np.asarray(task_ids),
        skill_ids=np.asarray(skill_ids),
        pair_ids=np.asarray(pair_ids),
        task_embeddings=task_embeddings,
        skill_embeddings=skill_embeddings,
        pair_task_indices=pair_task_indices,
        pair_skill_indices=pair_skill_indices,
        pair_features=pair_features,
        target_grade=target_grade,
        target_relevance=target_relevance,
        sample_weight=sample_weight,
        split_codes=split_codes,
    )
    metadata = {
        "schema_version": "skilldag_ncf.model_data.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_made": 0,
        "dataset_version": "task_skill_v1",
        "embedding_model": next(iter(models)),
        "embedding_dimension": int(task_embeddings.shape[1]),
        "task_count": len(task_ids),
        "skill_count": len(skill_ids),
        "pair_count": len(pair_rows),
        "feature_names": feature_names,
        "split_codes": DATASET_SPLITS,
        "target_grade_encoding": {"uncertain": -1, "negative": 0, "helpful": 1, "required": 2},
        "student_feature_contract": {
            "allowed": [
                "task_embedding",
                "skill_embedding",
                "locally_recomputed_task_skill_cosine_and_rank",
                "static_typed_graph_degree_features",
            ],
            "excluded_privileged_fields": list(PRIVILEGED_FIELDS),
        },
        "label_counts": dict(sorted(label_counts.items())),
        "provenance": {
            "dataset_dir": str(dataset_dir),
            "task_embedding_cache_dir": str(task_embedding_cache_dir),
            "skill_embedding_cache_path": str(skill_embedding_cache_path),
            "graph_path": str(graph_path),
            "dataset_files_sha256": {
                name: f"sha256:{_sha256_bytes((dataset_dir / name).read_bytes())}"
                for name in ("tasks.jsonl", "skills.jsonl", "pairs.jsonl", "splits.json")
            },
        },
        "outputs": {"arrays": str(arrays_path), "metadata": str(metadata_path)},
    }
    _atomic_json(metadata_path, metadata)
    return metadata


class TaskSkillDataset:
    """Small dependency-light dataset with a deterministic batch iterator."""

    def __init__(
        self,
        arrays_path: Path | str,
        *,
        split: str,
        trainable_only: bool = True,
    ) -> None:
        if split not in DATASET_SPLITS:
            raise ValueError(f"unknown dataset split: {split}")
        with np.load(Path(arrays_path).expanduser().resolve(), allow_pickle=False) as data:
            self.task_ids = data["task_ids"].copy()
            self.skill_ids = data["skill_ids"].copy()
            self.pair_ids = data["pair_ids"].copy()
            self.task_embeddings = data["task_embeddings"].copy()
            self.skill_embeddings = data["skill_embeddings"].copy()
            self.pair_task_indices = data["pair_task_indices"].copy()
            self.pair_skill_indices = data["pair_skill_indices"].copy()
            self.pair_features = data["pair_features"].copy()
            self.target_grade = data["target_grade"].copy()
            self.target_relevance = data["target_relevance"].copy()
            self.sample_weight = data["sample_weight"].copy()
            split_codes = data["split_codes"].copy()
        mask = split_codes == DATASET_SPLITS[split]
        if trainable_only:
            mask &= self.target_grade >= 0
        self.indices = np.flatnonzero(mask)
        self.split = split

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair_index = int(self.indices[index])
        task_index = int(self.pair_task_indices[pair_index])
        skill_index = int(self.pair_skill_indices[pair_index])
        return {
            "pair_index": pair_index,
            "pair_id": str(self.pair_ids[pair_index]),
            "task_id": str(self.task_ids[task_index]),
            "skill_id": str(self.skill_ids[skill_index]),
            "task_embedding": self.task_embeddings[task_index],
            "skill_embedding": self.skill_embeddings[skill_index],
            "pair_features": self.pair_features[pair_index],
            "target_grade": self.target_grade[pair_index],
            "target_relevance": self.target_relevance[pair_index],
            "sample_weight": self.sample_weight[pair_index],
        }

    def iter_batches(
        self, batch_size: int, *, shuffle: bool = False, seed: int = 0
    ) -> Iterator[dict[str, np.ndarray]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        order = np.arange(len(self.indices))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            selected = self.indices[order[start : start + batch_size]]
            task_indices = self.pair_task_indices[selected]
            skill_indices = self.pair_skill_indices[selected]
            yield {
                "pair_indices": selected,
                "task_embeddings": self.task_embeddings[task_indices],
                "skill_embeddings": self.skill_embeddings[skill_indices],
                "pair_features": self.pair_features[selected],
                "target_grade": self.target_grade[selected],
                "target_relevance": self.target_relevance[selected],
                "sample_weight": self.sample_weight[selected],
            }
