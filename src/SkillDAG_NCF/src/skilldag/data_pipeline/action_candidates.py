"""Build action-to-skill candidates from cached skill embeddings and graph edges."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


EmbeddingFunction = Callable[[list[str]], list[list[float]]]
WALKABLE_EDGE_TYPES = {"specializes", "composes_with", "depends_on", "similar_to"}
DEFAULT_EXCLUDED_ACTIONS = {"NoOp"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _split_camel_case(value: str) -> str:
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value).replace("_", " ")
    return " ".join(words.split()).lower()


def action_embedding_text(action: str) -> str:
    """Produce a deterministic query without manually assigning any skill label."""
    return f"ALFWorld expert-plan action: {_split_camel_case(action)}."


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else [0.0 for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"embedding dimension mismatch: {len(left)} != {len(right)}")
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    return float(sum(a * b for a, b in zip(left_norm, right_norm)))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def _collect_action_statistics(
    tasks: list[dict[str, Any]], excluded_actions: set[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    states: dict[str, dict[str, Any]] = {}
    for task in tasks:
        seen_in_task: set[str] = set()
        for step in task.get("plan_high_pddl", []):
            action = str(step.get("action", "")).strip()
            if not action:
                continue
            state = states.setdefault(
                action,
                {
                    "step_count": 0,
                    "task_count": 0,
                    "splits": Counter(),
                    "task_types": Counter(),
                    "argument_examples": [],
                    "_argument_keys": set(),
                },
            )
            state["step_count"] += 1
            args = step.get("args", []) or []
            args_key = json.dumps(args, ensure_ascii=False, sort_keys=True)
            if args_key not in state["_argument_keys"] and len(state["argument_examples"]) < 20:
                state["_argument_keys"].add(args_key)
                state["argument_examples"].append(args)
            if action not in seen_in_task:
                state["task_count"] += 1
                state["splits"][str(task.get("split", ""))] += 1
                state["task_types"][str(task.get("task_type", ""))] += 1
                seen_in_task.add(action)

    records = []
    for action, state in sorted(states.items()):
        records.append(
            {
                "action": action,
                "action_embedding_text": action_embedding_text(action),
                "excluded_from_candidates": action in excluded_actions,
                "exclusion_reason": "terminal/no-operation step" if action in excluded_actions else None,
                "step_count": state["step_count"],
                "task_count": state["task_count"],
                "counts_by_split": dict(sorted(state["splits"].items())),
                "counts_by_task_type": dict(sorted(state["task_types"].items())),
                "argument_examples": state["argument_examples"],
            }
        )
    return records, states


def _load_action_embeddings(
    actions: list[str],
    cache_path: Path,
    model: str,
    embed: EmbeddingFunction,
) -> tuple[dict[str, list[float]], int, int]:
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cache = loaded

    vectors: dict[str, list[float]] = {}
    missing: list[tuple[str, str, str]] = []
    for action in actions:
        text = action_embedding_text(action)
        text_hash = _sha256_text(text)[:16]
        entry = cache.get(action, {})
        if (
            entry.get("text_hash") == text_hash
            and entry.get("model") == model
            and isinstance(entry.get("embedding"), list)
        ):
            vectors[action] = entry["embedding"]
        else:
            missing.append((action, text, text_hash))

    if missing:
        generated = embed([text for _, text, _ in missing])
        if len(generated) != len(missing):
            raise ValueError(f"embedding API returned {len(generated)} vectors for {len(missing)} inputs")
        for (action, _, text_hash), vector in zip(missing, generated):
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"embedding API returned an invalid vector for {action}")
            vectors[action] = vector
            cache[action] = {"text_hash": text_hash, "model": model, "embedding": vector}
        _atomic_json(cache_path, cache)
    return vectors, len(actions) - len(missing), len(missing)


def _graph_adjacency(graph: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    adjacency: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in graph.get("edges", []) or []:
        edge_type = str(edge.get("type", ""))
        if edge_type not in WALKABLE_EDGE_TYPES:
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            adjacency[source].append({"skill_id": target, "via": edge_type, "reached_from": source})
            adjacency[target].append({"skill_id": source, "via": edge_type, "reached_from": target})
    return adjacency


def build_action_skill_candidates(
    tasks_path: Path | str,
    graph_path: Path | str,
    skill_embeddings_path: Path | str,
    output_root: Path | str,
    embedding_model: str,
    embed: EmbeddingFunction,
    *,
    top_k: int = 3,
    graph_seed_k: int = 0,
    excluded_actions: set[str] | None = None,
) -> dict[str, Any]:
    """Build candidate-only action/skill and task/skill tables.

    Existing skill vectors are mandatory and never recomputed here. Only
    missing unique action vectors are sent to ``embed``.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if graph_seed_k < 0 or graph_seed_k > top_k:
        raise ValueError("graph_seed_k must be between 0 and top_k")
    tasks_path = Path(tasks_path).expanduser().resolve()
    graph_path = Path(graph_path).expanduser().resolve()
    skill_embeddings_path = Path(skill_embeddings_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    excluded_actions = set(DEFAULT_EXCLUDED_ACTIONS if excluded_actions is None else excluded_actions)

    tasks = _load_jsonl(tasks_path)
    graph_raw = graph_path.read_bytes()
    graph = json.loads(graph_raw)
    graph_snapshot_id = f"sha256:{_sha256_bytes(graph_raw)}"
    skill_cache_raw = skill_embeddings_path.read_bytes()
    skill_cache = json.loads(skill_cache_raw)
    graph_skill_ids = set((graph.get("nodes", {}) or {}).keys())
    cache_skill_ids = set(skill_cache)
    if graph_skill_ids != cache_skill_ids:
        raise ValueError(
            "skill embedding cache does not match graph nodes: "
            f"missing={sorted(graph_skill_ids - cache_skill_ids)}, "
            f"extra={sorted(cache_skill_ids - graph_skill_ids)}"
        )
    wrong_models = sorted(
        skill_id
        for skill_id, entry in skill_cache.items()
        if entry.get("model") != embedding_model
    )
    if wrong_models:
        raise ValueError(
            f"{len(wrong_models)} cached skill vectors do not use model {embedding_model!r}"
        )
    skill_vectors = {skill_id: entry["embedding"] for skill_id, entry in skill_cache.items()}
    if not skill_vectors or any(not isinstance(vector, list) or not vector for vector in skill_vectors.values()):
        raise ValueError("skill embedding cache contains empty or invalid vectors")

    action_stats, _ = _collect_action_statistics(tasks, excluded_actions)
    candidate_actions = [
        item["action"] for item in action_stats if not item["excluded_from_candidates"]
    ]
    action_cache_path = output_root / "cache" / "action_embeddings.json"
    action_vectors, reused_action_count, embedded_action_count = _load_action_embeddings(
        candidate_actions, action_cache_path, embedding_model, embed
    )
    adjacency = _graph_adjacency(graph)

    action_candidates: list[dict[str, Any]] = []
    candidates_by_action: dict[str, list[dict[str, Any]]] = {}
    for action in candidate_actions:
        scored = sorted(
            (
                (skill_id, _cosine(action_vectors[action], vector))
                for skill_id, vector in skill_vectors.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        matches = scored[: min(top_k, len(scored))]
        merged: dict[str, dict[str, Any]] = {}
        for rank, (skill_id, score) in enumerate(matches, 1):
            merged[skill_id] = {
                "action": action,
                "skill_id": skill_id,
                "candidate_sources": ["action_embedding_match"],
                "cosine_score": score,
                "semantic_rank": rank,
                "graph_paths": [],
                "candidate_only": True,
                "graph_snapshot_id": graph_snapshot_id,
                "embedding_model": embedding_model,
            }
        # Graph expansion is opt-in. On the current dense ALFWorld graph, even
        # expanding only rank-1 approaches the whole 37-skill library after
        # candidates from a task's multiple actions are merged.
        for matched_skill, _ in matches[:graph_seed_k]:
            for neighbor in adjacency.get(matched_skill, []):
                skill_id = neighbor["skill_id"]
                record = merged.setdefault(
                    skill_id,
                    {
                        "action": action,
                        "skill_id": skill_id,
                        "candidate_sources": [],
                        "cosine_score": None,
                        "semantic_rank": None,
                        "graph_paths": [],
                        "candidate_only": True,
                        "graph_snapshot_id": graph_snapshot_id,
                        "embedding_model": embedding_model,
                    },
                )
                if "graph_one_hop" not in record["candidate_sources"]:
                    record["candidate_sources"].append("graph_one_hop")
                path = {
                    "reached_from": matched_skill,
                    "via": neighbor["via"],
                }
                if path not in record["graph_paths"]:
                    record["graph_paths"].append(path)
        records = sorted(
            merged.values(),
            key=lambda item: (
                item["semantic_rank"] is None,
                item["semantic_rank"] or 10**9,
                item["skill_id"],
            ),
        )
        candidates_by_action[action] = records
        action_candidates.extend(records)

    task_candidates: list[dict[str, Any]] = []
    tasks_without_candidates = 0
    per_task_counts: list[int] = []
    skill_exposure_counts: Counter[str] = Counter()
    for task in tasks:
        actions = []
        for step in task.get("plan_high_pddl", []):
            action = str(step.get("action", "")).strip()
            if action and action not in excluded_actions and action not in actions:
                actions.append(action)
        merged: dict[str, dict[str, Any]] = {}
        for action in actions:
            for candidate in candidates_by_action.get(action, []):
                skill_id = candidate["skill_id"]
                record = merged.setdefault(
                    skill_id,
                    {
                        "task_record_id": task["record_id"],
                        "source_task_id": task.get("source_task_id", ""),
                        "split": task.get("split", ""),
                        "task_type": task.get("task_type", ""),
                        "skill_id": skill_id,
                        "matched_actions": [],
                        "candidate_sources": [],
                        "best_cosine_score": None,
                        "best_semantic_rank": None,
                        "graph_snapshot_id": graph_snapshot_id,
                        "candidate_only": True,
                    },
                )
                if action not in record["matched_actions"]:
                    record["matched_actions"].append(action)
                for source in candidate["candidate_sources"]:
                    if source not in record["candidate_sources"]:
                        record["candidate_sources"].append(source)
                score = candidate["cosine_score"]
                if score is not None and (
                    record["best_cosine_score"] is None or score > record["best_cosine_score"]
                ):
                    record["best_cosine_score"] = score
                rank = candidate["semantic_rank"]
                if rank is not None and (
                    record["best_semantic_rank"] is None or rank < record["best_semantic_rank"]
                ):
                    record["best_semantic_rank"] = rank
        records = sorted(
            merged.values(),
            key=lambda item: (
                item["best_semantic_rank"] is None,
                item["best_semantic_rank"] or 10**9,
                -(item["best_cosine_score"] or -1.0),
                item["skill_id"],
            ),
        )
        if not records:
            tasks_without_candidates += 1
        per_task_counts.append(len(records))
        skill_exposure_counts.update(item["skill_id"] for item in records)
        task_candidates.extend(records)

    intermediate = output_root / "intermediate"
    reports = output_root / "reports"
    action_stats_path = reports / "action_statistics.json"
    action_candidates_path = intermediate / "action_skill_candidates.jsonl"
    task_candidates_path = intermediate / "task_skill_candidates.jsonl"
    _atomic_json(action_stats_path, action_stats)
    action_candidate_count = _atomic_jsonl(action_candidates_path, action_candidates)
    task_candidate_count = _atomic_jsonl(task_candidates_path, task_candidates)

    report = {
        "schema_version": "skilldag_ncf.action_candidates_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "labels_generated": 0,
        "configuration": {
            "top_k": top_k,
            "graph_seed_k": graph_seed_k,
            "graph_depth": 1,
            "walkable_edge_types": sorted(WALKABLE_EDGE_TYPES),
            "excluded_actions": sorted(excluded_actions),
            "embedding_model": embedding_model,
        },
        "embedding_usage": {
            "skill_vectors_reused": len(skill_vectors),
            "skill_vectors_computed": 0,
            "action_vectors_reused": reused_action_count,
            "action_vectors_computed": embedded_action_count,
            "embedding_inputs_sent": embedded_action_count,
        },
        "actions": {
            "unique_count": len(action_stats),
            "candidate_action_count": len(candidate_actions),
            "excluded_action_count": len(action_stats) - len(candidate_actions),
            "action_skill_candidate_count": action_candidate_count,
            "candidate_counts": {
                action: len(records) for action, records in sorted(candidates_by_action.items())
            },
        },
        "tasks": {
            "count": len(tasks),
            "task_skill_candidate_count": task_candidate_count,
            "tasks_without_candidates": tasks_without_candidates,
            "candidate_count_min": min(per_task_counts) if per_task_counts else 0,
            "candidate_count_max": max(per_task_counts) if per_task_counts else 0,
            "candidate_count_mean": (
                sum(per_task_counts) / len(per_task_counts) if per_task_counts else 0.0
            ),
        },
        "skills": {
            "count": len(skill_vectors),
            "task_candidate_exposure_counts": dict(sorted(skill_exposure_counts.items())),
        },
        "provenance": {
            "tasks_path": str(tasks_path),
            "tasks_sha256": f"sha256:{_sha256_bytes(tasks_path.read_bytes())}",
            "graph_path": str(graph_path),
            "graph_snapshot_id": graph_snapshot_id,
            "skill_embeddings_path": str(skill_embeddings_path),
            "skill_embeddings_sha256": f"sha256:{_sha256_bytes(skill_cache_raw)}",
        },
        "outputs": {
            "action_statistics": str(action_stats_path),
            "action_embeddings_cache": str(action_cache_path),
            "action_skill_candidates": str(action_candidates_path),
            "task_skill_candidates": str(task_candidates_path),
        },
    }
    report_path = reports / "candidate_coverage_report.json"
    report["outputs"]["report"] = str(report_path)
    _atomic_json(report_path, report)
    return report
