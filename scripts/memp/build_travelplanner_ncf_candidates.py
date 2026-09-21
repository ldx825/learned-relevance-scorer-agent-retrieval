#!/usr/bin/env python3
"""Build fixed-budget TravelPlanner Task-Context--Memory Judge candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_memory_embedding_candidates import embed_texts, normalized  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def cosine_memory_vectors(path: Path, expected_queries: list[str], model: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model:
        raise ValueError("Frozen memory-query embedding model mismatch")
    texts = payload.get("texts") or payload.get("queries") or []
    vectors = payload.get("vectors") or payload.get("embeddings") or []
    if texts[: len(expected_queries)] != expected_queries:
        raise ValueError("Frozen memory-query embedding text/order mismatch")
    result = np.asarray(vectors[: len(expected_queries)], dtype=np.float32)
    if len(result) != len(expected_queries):
        raise ValueError("Frozen memory-query embedding count mismatch")
    return result


def constraint_values(signature: dict[str, Any]) -> tuple[Any, ...]:
    cuisine = signature.get("cuisine")
    if isinstance(cuisine, list):
        cuisine = tuple(sorted(str(value) for value in cuisine))
    return (
        signature.get("house_rule"),
        cuisine,
        signature.get("room_type"),
        signature.get("transportation"),
    )


def structural_score(task: dict[str, Any], memory: dict[str, Any], view_type: str) -> float:
    score = 0.0
    score += 2.0 * (task["days"] == memory["days"])
    score += 2.0 * (task["visiting_city_number"] == memory["visiting_city_number"])
    score += 0.75 * (task["people_number"] == memory["people_number"])
    task_ppd = float(task["budget_per_person_day"])
    memory_ppd = float(memory["budget_per_person_day"])
    score += max(0.0, 1.0 - abs(math.log((task_ppd + 1.0) / (memory_ppd + 1.0))))
    task_constraints = constraint_values(task)
    memory_constraints = constraint_values(memory)
    for task_value, memory_value in zip(task_constraints, memory_constraints, strict=True):
        if task_value is None:
            continue
        if task_value == memory_value:
            score += 2.0
        elif memory_value is not None:
            score -= 2.0
    if view_type == "route_closure" and task.get("transportation") == memory.get("transportation"):
        score += 2.0
    if view_type == "lodging_feasibility":
        score += 1.0 * (task.get("room_type") == memory.get("room_type"))
        score += 1.0 * (task.get("house_rule") == memory.get("house_rule"))
    if view_type == "dining_coverage":
        left = set(task.get("cuisine") or [])
        right = set(memory.get("cuisine") or [])
        score += len(left & right)
    return score


def explicit_conflict(task: dict[str, Any], memory: dict[str, Any]) -> bool:
    for key in ("house_rule", "room_type", "transportation"):
        left, right = task.get(key), memory.get(key)
        if left is not None and right is not None and left != right:
            return True
    left_cuisine = set(task.get("cuisine") or [])
    right_cuisine = set(memory.get("cuisine") or [])
    return bool(left_cuisine and right_cuisine and not left_cuisine.issubset(right_cuisine))


def capability_score(required: list[str], capabilities: dict[str, bool]) -> float:
    if not required:
        return 0.0
    aliases = {
        "lodging_feasibility": ("minimum_nights", "party_capacity"),
        "dining_coverage": ("restaurant_uniqueness", "cuisine_coverage"),
        "attraction_coverage": ("attraction_uniqueness",),
        "explicit_constraints": ("house_rule", "room_type", "cuisine_coverage", "transport_restriction"),
        "budget_accounting": ("budget_accounting", "full_party_cost"),
        "non_repetition": ("restaurant_uniqueness", "attraction_uniqueness"),
    }
    expanded = []
    for name in required:
        expanded.append(name)
        expanded.extend(aliases.get(name, ()))
    return sum(float(capabilities.get(name, False)) for name in set(expanded)) / len(set(expanded))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-views", type=Path, required=True)
    parser.add_argument("--memory-profiles", type=Path, required=True)
    parser.add_argument("--memory-query-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.candidate_budget != 20:
        raise ValueError("The preregistered V2 candidate budget is exactly 20")

    views = load_jsonl(args.query_views)
    profiles = json.loads(args.memory_profiles.read_text(encoding="utf-8"))
    if len(views) != 405 or len(profiles) != 45:
        raise ValueError(f"Expected 405 views and 45 profiles, got {len(views)} and {len(profiles)}")
    if any(row.get("validation_or_test_used") is not False for row in views + profiles):
        raise ValueError("Unsafe evaluation-derived input")
    if [int(row["memory_id"]) for row in profiles] != list(range(45)):
        raise ValueError("Memory IDs must be dense and ordered")

    api_key = os.environ.get("SKILLDAG_EMBEDDING_API_KEY", "")
    api_base = os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1")
    view_vectors = embed_texts(
        [row["query_text"] for row in views],
        model=args.embedding_model,
        cache_path=args.output_dir / "embedding_cache" / "query_views.json",
        api_base=api_base,
        api_key=api_key,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    memory_vectors = cosine_memory_vectors(
        args.memory_query_embeddings,
        [row["memory_query"] for row in profiles],
        args.embedding_model,
    )
    scores = normalized(view_vectors) @ normalized(memory_vectors).T

    batches = []
    exposure: Counter[int] = Counter()
    source_exposure: Counter[str] = Counter()
    for task_id, view in enumerate(views):
        parent = int(view["parent_train_index"])
        required = list(view["required_capabilities"])
        cosine_order = np.argsort(-scores[task_id], kind="stable").tolist()
        cosine_rank = {int(memory_id): rank for rank, memory_id in enumerate(cosine_order, 1)}
        structured = sorted(
            range(45),
            key=lambda memory_id: (
                -structural_score(view["signature"], profiles[memory_id]["source_signature"], view["view_type"]),
                cosine_rank[memory_id],
                memory_id,
            ),
        )
        capability = sorted(
            range(45),
            key=lambda memory_id: (
                -capability_score(required, profiles[memory_id]["capabilities"]),
                -structural_score(view["signature"], profiles[memory_id]["source_signature"], view["view_type"]),
                cosine_rank[memory_id],
                memory_id,
            ),
        )
        hard = sorted(
            range(45),
            key=lambda memory_id: (
                not explicit_conflict(view["signature"], profiles[memory_id]["source_signature"])
                and capability_score(required, profiles[memory_id]["capabilities"]) >= 0.5,
                cosine_rank[memory_id],
                memory_id,
            ),
        )

        selected: list[int] = []
        sources: dict[int, set[str]] = {memory_id: set() for memory_id in range(45)}

        def take(order: list[int], count: int, source: str) -> None:
            added = 0
            for memory_id in order:
                sources[memory_id].add(source)
                if memory_id in selected:
                    continue
                selected.append(memory_id)
                added += 1
                if added == count:
                    return

        take([parent], 1, "same_train_source")
        take(cosine_order, 7, "query_cosine")
        take(structured, 6, "structure_match")
        take(capability, 3, "workflow_capability")
        take(hard, 3, "hard_negative")
        if len(selected) != args.candidate_budget:
            raise ValueError(f"Candidate budget mismatch for {view['view_id']}: {len(selected)}")

        candidates = []
        for memory_id in sorted(selected):
            profile = profiles[memory_id]
            candidate_sources = sorted(sources[memory_id])
            source_exposure.update(candidate_sources)
            exposure[memory_id] += 1
            candidates.append(
                {
                    "memory_id": memory_id,
                    "memory_source": profile["source"],
                    "memory_query": profile["memory_query"],
                    "workflow": profile["workflow"],
                    "candidate_sources": candidate_sources,
                    "query_cosine_rank": cosine_rank[memory_id],
                    "query_cosine_score": float(scores[task_id, memory_id]),
                    "structural_score": structural_score(
                        view["signature"], profile["source_signature"], view["view_type"]
                    ),
                    "capability_coverage": capability_score(required, profile["capabilities"]),
                    "explicit_conflict_seed": explicit_conflict(
                        view["signature"], profile["source_signature"]
                    ),
                    "memory_source_signature": profile["source_signature"],
                }
            )
        batches.append(
            {
                "schema_version": "memp.travelplanner.ncf_candidates.v2",
                "task_id": task_id,
                "view_id": view["view_id"],
                "family_id": view["family_id"],
                "parent_train_index": parent,
                "internal_split": view["model_split"],
                "view_type": view["view_type"],
                "task_query": view["task_query"],
                "focus_query": view["query_text"],
                "required_capabilities": required,
                "task_signature": view["signature"],
                "candidates": candidates,
                "seed_fields_hidden_from_judge": [
                    "candidate_sources",
                    "query_cosine_rank",
                    "query_cosine_score",
                    "structural_score",
                    "capability_coverage",
                    "explicit_conflict_seed",
                    "memory_source_signature",
                ],
                "validation_or_test_used": False,
            }
        )

    write_jsonl(args.output_dir / "judge_candidates.jsonl", batches)
    report = {
        "schema_version": "memp.travelplanner.ncf_candidates_report.v2",
        "query_context_count": len(batches),
        "memory_count": len(profiles),
        "candidate_budget": args.candidate_budget,
        "candidate_pair_count": sum(len(row["candidates"]) for row in batches),
        "candidate_source_target_counts_per_query": {
            "same_train_source": 1,
            "query_cosine": 7,
            "structure_match": 6,
            "workflow_capability": 3,
            "hard_negative": 3,
        },
        "candidate_source_exposure": dict(sorted(source_exposure.items())),
        "memory_exposure": {
            "min": min(exposure.values()),
            "mean": sum(exposure.values()) / len(exposure),
            "max": max(exposure.values()),
            "by_memory_id": {str(key): exposure[key] for key in range(45)},
        },
        "embedding_model": args.embedding_model,
        "split_counts": dict(Counter(row["internal_split"] for row in batches)),
        "view_counts": dict(Counter(row["view_type"] for row in batches)),
        "seed_fields_visible_to_judge": False,
        "validation_or_test_used": False,
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
