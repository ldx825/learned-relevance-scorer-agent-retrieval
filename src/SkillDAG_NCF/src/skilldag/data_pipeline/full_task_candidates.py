"""Scalable, resumable task-semantic candidates for a complete data split."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .action_candidates import (
    EmbeddingFunction,
    _atomic_json,
    _cosine,
    _load_jsonl,
    _sha256_bytes,
    _sha256_text,
)
from .task_semantic_pilot import _distribution


ProgressFunction = Callable[[int, int, int, int], None]


def _load_valid_entry(
    cache: dict[str, dict[str, Any]], record_id: str, text_hash: str, model: str
) -> list[float] | None:
    entry = cache.get(record_id, {})
    if (
        entry.get("text_hash") == text_hash
        and entry.get("model") == model
        and isinstance(entry.get("embedding"), list)
        and entry["embedding"]
    ):
        return entry["embedding"]
    return None


def build_full_split_task_candidates(
    tasks_path: Path | str,
    action_candidates_path: Path | str,
    graph_path: Path | str,
    skill_embeddings_path: Path | str,
    output_root: Path | str,
    embedding_model: str,
    embed: EmbeddingFunction,
    *,
    split: str = "train",
    task_top_k: int = 3,
    batch_size: int = 100,
    progress: ProgressFunction | None = None,
) -> dict[str, Any]:
    """Build task/action union candidates for every task in ``split``.

    Task embeddings are stored in deterministic shards and committed after
    every API batch so an interrupted run can resume without paying twice.
    """
    if task_top_k < 1 or batch_size < 1:
        raise ValueError("task_top_k and batch_size must be positive")
    tasks_path = Path(tasks_path).expanduser().resolve()
    action_candidates_path = Path(action_candidates_path).expanduser().resolve()
    graph_path = Path(graph_path).expanduser().resolve()
    skill_embeddings_path = Path(skill_embeddings_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    tasks = sorted(
        (task for task in _load_jsonl(tasks_path) if task.get("split") == split),
        key=lambda task: (task.get("task_type", ""), task["record_id"]),
    )
    if not tasks:
        raise ValueError(f"no tasks found for split {split!r}")
    task_ids = {task["record_id"] for task in tasks}

    action_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(action_candidates_path):
        if row.get("task_record_id") in task_ids:
            action_by_task[row["task_record_id"]].append(row)

    graph_raw = graph_path.read_bytes()
    graph = json.loads(graph_raw)
    graph_snapshot_id = f"sha256:{_sha256_bytes(graph_raw)}"
    skill_cache_raw = skill_embeddings_path.read_bytes()
    skill_cache = json.loads(skill_cache_raw)
    if set(skill_cache) != set((graph.get("nodes", {}) or {})):
        raise ValueError("skill embedding cache keys do not match graph nodes")
    if any(entry.get("model") != embedding_model for entry in skill_cache.values()):
        raise ValueError("skill embedding cache model does not match requested model")
    skill_vectors = {skill_id: entry["embedding"] for skill_id, entry in skill_cache.items()}

    legacy_path = output_root / "cache" / "pilot_task_embeddings.json"
    legacy_cache = json.loads(legacy_path.read_text(encoding="utf-8")) if legacy_path.exists() else {}
    shard_dir = output_root / "cache" / "task_embeddings" / split
    shard_dir.mkdir(parents=True, exist_ok=True)

    intermediate = output_root / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    semantic_path = intermediate / f"{split}_task_semantic_candidates.jsonl"
    combined_path = intermediate / f"{split}_combined_candidates.jsonl"
    semantic_tmp = semantic_path.with_name(f".{semantic_path.name}.tmp-{os.getpid()}")
    combined_tmp = combined_path.with_name(f".{combined_path.name}.tmp-{os.getpid()}")

    computed_count = 0
    shard_reused_count = 0
    legacy_reused_count = 0
    semantic_pair_count = 0
    combined_pair_count = 0
    task_only_skills: set[str] = set()
    action_only_skills: set[str] = set()
    combined_skills: set[str] = set()
    task_counts: list[int] = []
    action_counts: list[int] = []
    combined_counts: list[int] = []
    overlap_counts: list[int] = []
    counts_by_type: Counter[str] = Counter()
    total_batches = (len(tasks) + batch_size - 1) // batch_size

    try:
        with semantic_tmp.open("w", encoding="utf-8") as semantic_handle, combined_tmp.open(
            "w", encoding="utf-8"
        ) as combined_handle:
            for batch_index in range(total_batches):
                batch = tasks[batch_index * batch_size : (batch_index + 1) * batch_size]
                shard_path = shard_dir / f"shard_{batch_index:05d}.json"
                shard: dict[str, dict[str, Any]] = {}
                if shard_path.exists():
                    loaded = json.loads(shard_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        shard = loaded
                vectors: dict[str, list[float]] = {}
                missing: list[tuple[dict[str, Any], str, str]] = []
                for task in batch:
                    record_id = task["record_id"]
                    text = str(task.get("task_text", "")).strip()
                    if not text:
                        raise ValueError(f"task has empty task_text: {record_id}")
                    text_hash = _sha256_text(text)[:16]
                    vector = _load_valid_entry(shard, record_id, text_hash, embedding_model)
                    if vector is not None:
                        vectors[record_id] = vector
                        shard_reused_count += 1
                        continue
                    vector = _load_valid_entry(legacy_cache, record_id, text_hash, embedding_model)
                    if vector is not None:
                        vectors[record_id] = vector
                        shard[record_id] = {
                            "text_hash": text_hash,
                            "model": embedding_model,
                            "embedding": vector,
                        }
                        legacy_reused_count += 1
                    else:
                        missing.append((task, text, text_hash))
                if missing:
                    generated = embed([text for _, text, _ in missing])
                    if len(generated) != len(missing):
                        raise ValueError(
                            f"embedding API returned {len(generated)} vectors for {len(missing)} inputs"
                        )
                    for (task, _, text_hash), vector in zip(missing, generated):
                        if not isinstance(vector, list) or not vector:
                            raise ValueError(f"invalid task embedding for {task['record_id']}")
                        vectors[task["record_id"]] = vector
                        shard[task["record_id"]] = {
                            "text_hash": text_hash,
                            "model": embedding_model,
                            "embedding": vector,
                        }
                        computed_count += 1
                # Commit this batch before candidate generation or proceeding.
                _atomic_json(shard_path, shard)

                for task in batch:
                    record_id = task["record_id"]
                    counts_by_type[str(task.get("task_type", ""))] += 1
                    scored = sorted(
                        (
                            (skill_id, _cosine(vectors[record_id], vector))
                            for skill_id, vector in skill_vectors.items()
                        ),
                        key=lambda item: (-item[1], item[0]),
                    )[: min(task_top_k, len(skill_vectors))]
                    semantic_by_skill: dict[str, dict[str, Any]] = {}
                    for rank, (skill_id, score) in enumerate(scored, 1):
                        row = {
                            "task_record_id": record_id,
                            "source_task_id": task.get("source_task_id", ""),
                            "split": split,
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
                        semantic_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        semantic_pair_count += 1

                    action_by_skill = {
                        row["skill_id"]: row for row in action_by_task.get(record_id, [])
                    }
                    semantic_ids = set(semantic_by_skill)
                    action_ids = set(action_by_skill)
                    union_ids = semantic_ids | action_ids
                    task_only_skills.update(semantic_ids)
                    action_only_skills.update(action_ids)
                    combined_skills.update(union_ids)
                    task_counts.append(len(semantic_ids))
                    action_counts.append(len(action_ids))
                    combined_counts.append(len(union_ids))
                    overlap_counts.append(len(semantic_ids & action_ids))
                    for skill_id in sorted(union_ids):
                        semantic = semantic_by_skill.get(skill_id)
                        action = action_by_skill.get(skill_id)
                        sources = ["task_embedding_match"] if semantic else []
                        if action:
                            sources.extend(
                                source
                                for source in action.get("candidate_sources", [])
                                if source not in sources
                            )
                        row = {
                            "task_record_id": record_id,
                            "source_task_id": task.get("source_task_id", ""),
                            "split": split,
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
                        combined_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        combined_pair_count += 1
                semantic_handle.flush()
                combined_handle.flush()
                if progress:
                    progress(batch_index + 1, total_batches, len(batch), len(missing))
        semantic_tmp.replace(semantic_path)
        combined_tmp.replace(combined_path)
    finally:
        if semantic_tmp.exists():
            semantic_tmp.unlink()
        if combined_tmp.exists():
            combined_tmp.unlink()

    report = {
        "schema_version": "skilldag_ncf.full_task_candidates_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "labels_generated": 0,
        "configuration": {
            "split": split,
            "task_top_k": task_top_k,
            "graph_expansion": False,
            "embedding_model": embedding_model,
            "embedding_batch_size": batch_size,
        },
        "tasks": {
            "count": len(tasks),
            "counts_by_task_type": dict(sorted(counts_by_type.items())),
        },
        "embedding_usage": {
            "skill_vectors_reused": len(skill_vectors),
            "skill_vectors_computed": 0,
            "task_vectors_reused_from_shards": shard_reused_count,
            "task_vectors_reused_from_pilot": legacy_reused_count,
            "task_vectors_computed": computed_count,
            "embedding_inputs_sent": computed_count,
            "cache_shard_count": total_batches,
        },
        "candidates": {
            "task_semantic_pair_count": semantic_pair_count,
            "combined_pair_count": combined_pair_count,
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
            "task_embedding_shards": str(shard_dir),
            "task_semantic_candidates": str(semantic_path),
            "combined_candidates": str(combined_path),
        },
    }
    report_path = output_root / "reports" / f"{split}_task_candidates_report.json"
    report["outputs"]["report"] = str(report_path)
    _atomic_json(report_path, report)
    return report
