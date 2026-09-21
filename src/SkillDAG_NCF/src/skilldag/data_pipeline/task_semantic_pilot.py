"""Small, stratified pilot for task-text semantic candidate retrieval."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_candidates import (
    EmbeddingFunction,
    _atomic_json,
    _atomic_jsonl,
    _cosine,
    _load_jsonl,
    _sha256_bytes,
    _sha256_text,
)


PILOT_SEED = "skilldag-ncf-task-semantic-pilot-v1"


def _select_stratified(
    tasks: list[dict[str, Any]], split: str, samples_per_task_type: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task.get("split") == split:
            groups[str(task.get("task_type", ""))].append(task)
    selected = []
    for task_type in sorted(groups):
        ranked = sorted(
            groups[task_type],
            key=lambda item: hashlib.sha256(
                f"{PILOT_SEED}:{item['record_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[:samples_per_task_type])
    return sorted(selected, key=lambda item: (item["task_type"], item["record_id"]))


def build_stratified_selection(
    tasks_path: Path | str,
    output_path: Path | str,
    *,
    split: str = "train",
    samples_per_task_type: int = 100,
    seed: str = PILOT_SEED,
) -> dict[str, Any]:
    """Create a deterministic, nested task selection for Judge collection.

    Keeping the pilot seed makes the first 10 tasks of every task type a
    subset of a later 100-per-type collection, so existing Judge responses
    can be reused safely.
    """
    if samples_per_task_type < 1:
        raise ValueError("samples_per_task_type must be positive")
    tasks_path = Path(tasks_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    tasks = _load_jsonl(tasks_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task.get("split") == split:
            groups[str(task.get("task_type", ""))].append(task)
    if not groups:
        raise ValueError(f"no tasks found for split {split!r}")

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for task_type in sorted(groups):
        ranked = sorted(
            groups[task_type],
            key=lambda item: hashlib.sha256(
                f"{seed}:{item['record_id']}".encode("utf-8")
            ).hexdigest(),
        )
        chosen = ranked[:samples_per_task_type]
        selected.extend(chosen)
        counts[task_type] = len(chosen)
    selected.sort(key=lambda item: (item["task_type"], item["record_id"]))
    payload = {
        "schema_version": "skilldag_ncf.stratified_selection.v1",
        "seed": seed,
        "split": split,
        "samples_per_task_type": samples_per_task_type,
        "task_count": len(selected),
        "counts_by_task_type": counts,
        "record_ids": [task["record_id"] for task in selected],
    }
    _atomic_json(output_path, payload)
    return payload


def _load_task_embeddings(
    tasks: list[dict[str, Any]],
    cache_path: Path,
    model: str,
    embed: EmbeddingFunction,
) -> tuple[dict[str, list[float]], int, int]:
    # Reuse the generic, hash/model-aware cache helper by treating record IDs
    # as keys and task text as their embedding text.
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cache = loaded
    vectors: dict[str, list[float]] = {}
    missing: list[tuple[str, str, str]] = []
    for task in tasks:
        record_id = task["record_id"]
        text = str(task.get("task_text", "")).strip()
        if not text:
            raise ValueError(f"pilot task has empty task_text: {record_id}")
        text_hash = _sha256_text(text)[:16]
        entry = cache.get(record_id, {})
        if (
            entry.get("text_hash") == text_hash
            and entry.get("model") == model
            and isinstance(entry.get("embedding"), list)
        ):
            vectors[record_id] = entry["embedding"]
        else:
            missing.append((record_id, text, text_hash))
    if missing:
        generated = embed([text for _, text, _ in missing])
        if len(generated) != len(missing):
            raise ValueError(f"embedding API returned {len(generated)} vectors for {len(missing)} inputs")
        for (record_id, _, text_hash), vector in zip(missing, generated):
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"invalid task embedding for {record_id}")
            vectors[record_id] = vector
            cache[record_id] = {
                "text_hash": text_hash,
                "model": model,
                "embedding": vector,
            }
        _atomic_json(cache_path, cache)
    return vectors, len(tasks) - len(missing), len(missing)


def _distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "mean": sum(values) / len(values) if values else 0.0,
    }


def build_task_semantic_pilot(
    tasks_path: Path | str,
    action_candidates_path: Path | str,
    graph_path: Path | str,
    skill_embeddings_path: Path | str,
    output_root: Path | str,
    embedding_model: str,
    embed: EmbeddingFunction,
    *,
    split: str = "train",
    samples_per_task_type: int = 10,
    task_top_k: int = 3,
) -> dict[str, Any]:
    """Run a small task-text retrieval pilot without graph expansion or labels."""
    if samples_per_task_type < 1 or task_top_k < 1:
        raise ValueError("samples_per_task_type and task_top_k must be positive")
    tasks_path = Path(tasks_path).expanduser().resolve()
    action_candidates_path = Path(action_candidates_path).expanduser().resolve()
    graph_path = Path(graph_path).expanduser().resolve()
    skill_embeddings_path = Path(skill_embeddings_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    all_tasks = _load_jsonl(tasks_path)
    selected = _select_stratified(all_tasks, split, samples_per_task_type)
    if not selected:
        raise ValueError(f"no tasks selected for split {split!r}")
    selected_ids = {task["record_id"] for task in selected}

    graph_raw = graph_path.read_bytes()
    graph = json.loads(graph_raw)
    graph_snapshot_id = f"sha256:{_sha256_bytes(graph_raw)}"
    skill_cache_raw = skill_embeddings_path.read_bytes()
    skill_cache = json.loads(skill_cache_raw)
    graph_skill_ids = set((graph.get("nodes", {}) or {}).keys())
    cache_skill_ids = set(skill_cache)
    if graph_skill_ids != cache_skill_ids:
        raise ValueError("skill embedding cache keys do not match graph nodes")
    wrong_models = [sid for sid, entry in skill_cache.items() if entry.get("model") != embedding_model]
    if wrong_models:
        raise ValueError(f"{len(wrong_models)} skill vectors use a different embedding model")
    skill_vectors = {sid: entry["embedding"] for sid, entry in skill_cache.items()}

    task_cache_path = output_root / "cache" / "pilot_task_embeddings.json"
    task_vectors, reused_count, computed_count = _load_task_embeddings(
        selected, task_cache_path, embedding_model, embed
    )

    action_rows = _load_jsonl(action_candidates_path)
    action_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        if row.get("task_record_id") in selected_ids:
            action_by_task[row["task_record_id"]].append(row)

    task_semantic_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    task_only_skills: set[str] = set()
    action_only_skills: set[str] = set()
    combined_skills: set[str] = set()
    task_counts: list[int] = []
    action_counts: list[int] = []
    combined_counts: list[int] = []
    overlap_counts: list[int] = []
    counts_by_type: Counter[str] = Counter()

    for task in selected:
        record_id = task["record_id"]
        counts_by_type[task["task_type"]] += 1
        scored = sorted(
            (
                (skill_id, _cosine(task_vectors[record_id], vector))
                for skill_id, vector in skill_vectors.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )[: min(task_top_k, len(skill_vectors))]
        semantic_by_skill: dict[str, dict[str, Any]] = {}
        for rank, (skill_id, score) in enumerate(scored, 1):
            row = {
                "task_record_id": record_id,
                "source_task_id": task.get("source_task_id", ""),
                "split": task.get("split", ""),
                "task_type": task.get("task_type", ""),
                "task_text": task.get("task_text", ""),
                "skill_id": skill_id,
                "candidate_source": "task_embedding_match",
                "task_cosine_score": score,
                "task_semantic_rank": rank,
                "embedding_model": embedding_model,
                "graph_snapshot_id": graph_snapshot_id,
                "candidate_only": True,
            }
            semantic_by_skill[skill_id] = row
            task_semantic_rows.append(row)

        action_by_skill = {row["skill_id"]: row for row in action_by_task.get(record_id, [])}
        task_ids = set(semantic_by_skill)
        action_ids = set(action_by_skill)
        combined_ids = task_ids | action_ids
        task_only_skills.update(task_ids)
        action_only_skills.update(action_ids)
        combined_skills.update(combined_ids)
        task_counts.append(len(task_ids))
        action_counts.append(len(action_ids))
        combined_counts.append(len(combined_ids))
        overlap_counts.append(len(task_ids & action_ids))

        for skill_id in sorted(combined_ids):
            semantic = semantic_by_skill.get(skill_id)
            action = action_by_skill.get(skill_id)
            sources = []
            if semantic:
                sources.append("task_embedding_match")
            if action:
                sources.extend(
                    source for source in action.get("candidate_sources", []) if source not in sources
                )
            combined_rows.append(
                {
                    "task_record_id": record_id,
                    "source_task_id": task.get("source_task_id", ""),
                    "split": task.get("split", ""),
                    "task_type": task.get("task_type", ""),
                    "task_text": task.get("task_text", ""),
                    "skill_id": skill_id,
                    "candidate_sources": sources,
                    "task_cosine_score": semantic.get("task_cosine_score") if semantic else None,
                    "task_semantic_rank": semantic.get("task_semantic_rank") if semantic else None,
                    "action_cosine_score": action.get("best_cosine_score") if action else None,
                    "action_semantic_rank": action.get("best_semantic_rank") if action else None,
                    "matched_actions": action.get("matched_actions", []) if action else [],
                    "graph_snapshot_id": graph_snapshot_id,
                    "candidate_only": True,
                }
            )

    intermediate = output_root / "intermediate"
    reports = output_root / "reports"
    selection_path = reports / "task_semantic_pilot_selection.json"
    semantic_path = intermediate / "pilot_task_semantic_candidates.jsonl"
    combined_path = intermediate / "pilot_combined_candidates.jsonl"
    _atomic_json(
        selection_path,
        {
            "seed": PILOT_SEED,
            "split": split,
            "samples_per_task_type": samples_per_task_type,
            "record_ids": [task["record_id"] for task in selected],
        },
    )
    semantic_count = _atomic_jsonl(semantic_path, task_semantic_rows)
    combined_count = _atomic_jsonl(combined_path, combined_rows)

    report = {
        "schema_version": "skilldag_ncf.task_semantic_pilot_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "labels_generated": 0,
        "configuration": {
            "split": split,
            "samples_per_task_type": samples_per_task_type,
            "selection_seed": PILOT_SEED,
            "task_top_k": task_top_k,
            "graph_expansion": False,
            "embedding_model": embedding_model,
        },
        "sample": {
            "task_count": len(selected),
            "counts_by_task_type": dict(sorted(counts_by_type.items())),
        },
        "embedding_usage": {
            "skill_vectors_reused": len(skill_vectors),
            "skill_vectors_computed": 0,
            "task_vectors_reused": reused_count,
            "task_vectors_computed": computed_count,
            "embedding_inputs_sent": computed_count,
        },
        "candidates": {
            "task_semantic_pair_count": semantic_count,
            "combined_pair_count": combined_count,
            "task_semantic_per_task": _distribution(task_counts),
            "action_per_task": _distribution(action_counts),
            "combined_per_task": _distribution(combined_counts),
            "task_action_overlap_per_task": _distribution(overlap_counts),
        },
        "skill_coverage": {
            "library_size": len(skill_vectors),
            "action_view_count": len(action_only_skills),
            "task_view_count": len(task_only_skills),
            "combined_count": len(combined_skills),
            "new_skills_from_task_view": sorted(task_only_skills - action_only_skills),
            "still_uncovered": sorted(set(skill_vectors) - combined_skills),
        },
        "provenance": {
            "tasks_path": str(tasks_path),
            "tasks_sha256": f"sha256:{_sha256_bytes(tasks_path.read_bytes())}",
            "action_candidates_path": str(action_candidates_path),
            "action_candidates_sha256": f"sha256:{_sha256_bytes(action_candidates_path.read_bytes())}",
            "graph_path": str(graph_path),
            "graph_snapshot_id": graph_snapshot_id,
            "skill_embeddings_path": str(skill_embeddings_path),
            "skill_embeddings_sha256": f"sha256:{_sha256_bytes(skill_cache_raw)}",
        },
        "outputs": {
            "selection": str(selection_path),
            "task_embeddings_cache": str(task_cache_path),
            "task_semantic_candidates": str(semantic_path),
            "combined_candidates": str(combined_path),
        },
    }
    report_path = reports / "task_semantic_pilot_report.json"
    report["outputs"]["report"] = str(report_path)
    _atomic_json(report_path, report)
    return report
