#!/usr/bin/env python3
"""Build fair phase-composed TravelPlanner retrieval for cosine or Content-NeuMF.

Both methods see the same 519 train-derived operation views, the same five
visible-task phase queries, and the same two-operations-per-phase budget.  The
script never reads official labels, rewards, or trajectories.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NCF_SRC = ROOT / "src/GraphOfSkills_NCF"
for value in (str(HERE), str(NCF_SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from build_memory_embedding_candidates import embed_texts  # noqa: E402
from build_travelplanner_atomic_phase_pairs import PHASES  # noqa: E402
from travelplanner_atomic_memory import render_dynamic_memory  # noqa: E402
from travelplanner_phase_query_contract import (  # noqa: E402
    QUERY_CONTRACTS,
    build_phase_query,
)
from gos.ncf.models import NCFConfig, NeuMF  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def task_signature(row: dict[str, str]) -> dict[str, Any]:
    constraints = ast.literal_eval(row["local_constraint"])
    days, people = int(row["days"]), int(row["people_number"])
    return {
        "level": row["level"],
        "days": days,
        "visiting_city_number": int(row["visiting_city_number"]),
        "people_number": people,
        "budget": int(row["budget"]),
        "budget_per_person_day": int(row["budget"]) / (people * days),
        "house_rule": constraints.get("house rule"),
        "cuisine": constraints.get("cuisine"),
        "room_type": constraints.get("room type"),
        "transportation": constraints.get("transportation"),
        "active_constraint_count": sum(value is not None for value in constraints.values()),
    }


def phase_queries(
    task_query: str,
    signature: dict[str, Any],
    query_contract: str = "v1",
) -> list[dict[str, str]]:
    return [
        {
            "phase_key": str(phase["key"]),
            "query": build_phase_query(
                task_query, phase, signature, contract=query_contract
            ),
        }
        for phase in PHASES
    ]


def compatible(signature: dict[str, Any], operation: dict[str, Any]) -> bool:
    """Gate conditional operations by field presence, not exact source value.

    A train Memory teaches how to enforce a constraint type.  Requiring its
    concrete cuisine/room/transport value to equal an unseen task would remove
    the procedural operation before either retriever can rank it.
    """
    field = operation.get("applicability_field")
    if not field:
        return True
    required = signature.get(field)
    return bool(required)


def unique_key_order(
    scores: np.ndarray,
    allowed: np.ndarray,
    operations: list[dict[str, Any]],
    candidate_budget: int,
) -> list[int]:
    ordered = allowed[np.argsort(-scores[allowed], kind="stable")]
    if candidate_budget > 0:
        ordered = ordered[:candidate_budget]
    seen: set[str] = set()
    result = []
    for index in ordered:
        key = str(operations[int(index)]["operation_key"])
        if key in seen:
            continue
        seen.add(key)
        result.append(int(index))
    return result


def diverse_cosine_pool(
    cosine: np.ndarray,
    allowed: np.ndarray,
    operations: list[dict[str, Any]],
    budget: int,
) -> np.ndarray:
    """Keep cosine recall size fixed while preventing text variants monopolizing it."""
    ordered = allowed[np.argsort(-cosine[allowed], kind="stable")]
    if budget <= 0 or budget >= len(ordered):
        return ordered
    first_by_key: list[int] = []
    seen_keys: set[str] = set()
    for index in ordered:
        key = str(operations[int(index)]["operation_key"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        first_by_key.append(int(index))
        if len(first_by_key) == budget:
            return np.asarray(first_by_key, dtype=np.int32)
    selected = set(first_by_key)
    remainder = [int(index) for index in ordered if int(index) not in selected]
    return np.asarray((first_by_key + remainder)[:budget], dtype=np.int32)


def phase_role_keys(phase: dict[str, Any], signature: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return visible-task anchor and complementary operation roles.

    Roles are declared in the train-only phase ontology.  Conditional roles
    enter the pool only when their user-visible constraint is active.
    """
    anchors = {str(value) for value in phase.get("anchor", ())}
    complementary = {
        str(value)
        for field in ("primary", "support")
        for value in phase.get(field, ())
    } - anchors
    for field, operation_key in phase.get("conditional", {}).items():
        if signature.get(field):
            complementary.add(str(operation_key))
    return anchors, complementary


def select_role_complementary(
    ranked: list[int],
    operations: list[dict[str, Any]],
    phase: dict[str, Any],
    signature: dict[str, Any],
    chosen_keys: set[str],
    reserved_keys: set[str],
    per_phase: int,
) -> list[int]:
    """Select one phase anchor and complementary roles without score fusion.

    The scorer still decides which parent-specific operation view wins.  The
    decoder only prevents individually high-scoring primary operations from
    consuming every slot needed for route/evidence closure.
    """
    if per_phase != 2:
        raise ValueError("role_complementary currently requires --per-phase 2")
    anchor_keys, complementary_keys = phase_role_keys(phase, signature)
    required = tuple(str(value) for value in phase.get("required_complement", ()))
    active_conditional = tuple(
        str(operation_key)
        for field, operation_key in phase.get("conditional", {}).items()
        if signature.get(field)
    )

    def best_index(eligible: set[str]) -> int | None:
        for index in ranked:
            key = str(operations[index]["operation_key"])
            if key in eligible and key not in chosen_keys:
                return index
        return None

    anchor = best_index(anchor_keys)
    if anchor is None:
        raise ValueError(f"no anchor operation for phase={phase['key']}")
    selected = [anchor]
    local_chosen = chosen_keys | {str(operations[anchor]["operation_key"])}

    complement = None
    for required_key in required:
        complement = next(
            (
                index for index in ranked
                if str(operations[index]["operation_key"]) == required_key
                and required_key not in local_chosen
            ),
            None,
        )
        if complement is not None:
            break
    if complement is None:
        complement = best_index(set(active_conditional) - local_chosen)
    if complement is None:
        eligible = complementary_keys - local_chosen - reserved_keys
        complement = best_index(eligible)
    if complement is None:
        raise ValueError(f"no complementary operation for phase={phase['key']}")
    selected.append(complement)
    return selected


def score_neumf_queries(
    model: NeuMF,
    query_embeddings: np.ndarray,
    operation_embeddings: np.ndarray,
    query_batch_size: int,
) -> np.ndarray:
    """Compute the same query-operation logits with fewer CPU forward calls."""
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    output = np.empty(
        (len(query_embeddings), len(operation_embeddings)), dtype=np.float32
    )
    with torch.no_grad():
        for start in range(0, len(query_embeddings), query_batch_size):
            queries = query_embeddings[start : start + query_batch_size]
            task_batch = np.repeat(
                queries[:, None, :], len(operation_embeddings), axis=1
            ).reshape(-1, queries.shape[1])
            memory_batch = np.tile(operation_embeddings, (len(queries), 1))
            scores = model(
                torch.from_numpy(task_batch), torch.from_numpy(memory_batch)
            ).numpy()
            output[start : start + len(queries)] = scores.reshape(
                len(queries), len(operation_embeddings)
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--indices",
        help="Optional comma-separated split indices; overrides --start/--count.",
    )
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reranker", choices=("cosine", "neumf"), required=True)
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=0,
        help="Cosine shallow pool per phase; 0 lets the selected scorer rank all compatible views.",
    )
    parser.add_argument("--per-phase", type=int, default=2)
    parser.add_argument(
        "--selection-policy",
        choices=("topk", "role_complementary"),
        default="topk",
        help="Use raw per-phase Top-K or the train-declared anchor/complement contract.",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--query-contract",
        choices=QUERY_CONTRACTS,
        default="v1",
        help="Shared train/inference phase-query surface; v1 preserves prior runs.",
    )
    parser.add_argument("--inference-query-batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.indices:
        indices = [int(value.strip()) for value in args.indices.split(",") if value.strip()]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("--indices must contain unique split indices")
        if any(index < 0 or index >= len(rows) for index in indices):
            raise ValueError(f"indices must be in [0, {len(rows)})")
    else:
        indices = list(range(args.start, min(args.start + args.count, len(rows))))
    operations = read_jsonl(args.operations)
    if not indices or len(operations) != 519:
        raise ValueError("empty task slice or unexpected operation bank")
    if any(row.get("validation_or_test_used") is not False for row in operations):
        raise ValueError("operation bank is not train-only")

    tasks = []
    for index in indices:
        query = rows[index]["query"]
        signature = task_signature(rows[index])
        tasks.append((
            index,
            query,
            signature,
            phase_queries(query, signature, args.query_contract),
        ))
    texts = [phase["query"] for _, _, _, phases in tasks for phase in phases]
    if args.dry_run:
        print(json.dumps({
            "task_count": len(tasks),
            "phase_query_count": len(texts),
            "phase_count_per_task": len(PHASES),
            "operation_view_count": len(operations),
            "per_phase": args.per_phase,
            "final_operation_budget": len(PHASES) * args.per_phase,
            "candidate_budget": args.candidate_budget or len(operations),
            "reranker": args.reranker,
            "selection_policy": args.selection_policy,
            "query_contract": args.query_contract,
            "official_gold_reward_or_trajectory_used": False,
        }, indent=2))
        return 0

    vectors = embed_texts(
        texts,
        model=args.embedding_model,
        cache_path=args.cache,
        api_base=os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1"),
        api_key=os.environ.get("SKILLDAG_EMBEDDING_API_KEY", ""),
        batch_size=64,
        timeout=180,
    ).astype(np.float32)
    with np.load(args.arrays, allow_pickle=False) as payload:
        operation_embeddings = payload["memory_embeddings"].copy().astype(np.float32)
        operation_ids = [str(value) for value in payload["operation_ids"]]
    if operation_ids != [str(row["operation_id"]) for row in operations]:
        raise ValueError("operation embedding ordering mismatch")

    model = None
    if args.reranker == "neumf":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for NeuMF")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = NeuMF(NCFConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    op_norm = operation_embeddings / np.maximum(
        np.linalg.norm(operation_embeddings, axis=1, keepdims=True), 1e-12
    )
    neumf_scores = (
        score_neumf_queries(
            model, vectors, operation_embeddings, args.inference_query_batch_size
        )
        if model is not None
        else None
    )

    reserved_keys = {
        str(value)
        for phase in PHASES
        for field in ("anchor", "required_complement")
        for value in phase.get(field, ())
    }
    retrievals, cursor = [], 0
    for task_index, task_query, signature, phases in tasks:
        chosen_by_key: dict[str, dict[str, Any]] = {}
        diagnostics = []
        for phase in phases:
            phase_index = cursor
            vector = vectors[phase_index]
            cursor += 1
            cosine = op_norm @ (vector / max(float(np.linalg.norm(vector)), 1e-12))
            allowed = np.asarray([
                index for index, operation in enumerate(operations)
                if compatible(signature, operation)
            ], dtype=np.int32)
            if model is None:
                ranking_scores = cosine
            else:
                assert neumf_scores is not None
                ranking_scores = neumf_scores[phase_index]
            # A positive candidate budget means cosine recall followed by the
            # chosen scorer.  Zero means direct full-bank cosine or NeuMF.
            candidate_order = (
                diverse_cosine_pool(cosine, allowed, operations, args.candidate_budget)
                if args.candidate_budget > 0
                else allowed[np.argsort(-ranking_scores[allowed], kind="stable")]
            )
            ranked = unique_key_order(
                ranking_scores,
                candidate_order,
                operations,
                candidate_budget=0,
            )
            if args.selection_policy == "role_complementary":
                phase_config = next(row for row in PHASES if row["key"] == phase["phase_key"])
                phase_selected = select_role_complementary(
                    ranked,
                    operations,
                    phase_config,
                    signature,
                    set(chosen_by_key),
                    reserved_keys,
                    args.per_phase,
                )
            else:
                phase_selected = []
                for operation_index in ranked:
                    key = str(operations[operation_index]["operation_key"])
                    if key in chosen_by_key:
                        continue
                    phase_selected.append(operation_index)
                    if len(phase_selected) == args.per_phase:
                        break
            if len(phase_selected) != args.per_phase:
                raise ValueError(f"cannot satisfy phase budget for task={task_index} phase={phase['phase_key']}")
            for operation_index in phase_selected:
                key = str(operations[operation_index]["operation_key"])
                selected = dict(operations[operation_index], score=float(ranking_scores[operation_index]))
                field = selected.get("applicability_field")
                if field:
                    selected["source_applicability_value"] = selected.get("applicability_value")
                    selected["applicability_value"] = signature.get(field)
                chosen_by_key[key] = selected
            diagnostics.append({
                "phase_key": phase["phase_key"],
                "phase_query": phase["query"],
                "selected": [
                    {
                        "operation_id": operations[index]["operation_id"],
                        "operation_key": operations[index]["operation_key"],
                        "parent_memory_id": operations[index]["parent_memory_id"],
                        "score": float(ranking_scores[index]),
                        "cosine_score": float(cosine[index]),
                    }
                    for index in phase_selected
                ],
            })
        selected_operations = list(chosen_by_key.values())
        expected = len(PHASES) * args.per_phase
        if len(selected_operations) != expected:
            raise AssertionError(f"expected {expected} globally unique operations")
        retrievals.append({
            "test_index": task_index,
            "query": task_query,
            "dynamic_memory_text": render_dynamic_memory(task_query, selected_operations),
            "selected_operations": diagnostics,
            "retriever": args.reranker,
            "candidate_budget_per_phase": args.candidate_budget or len(operations),
            "final_operation_budget": expected,
            "gold_reward_or_trajectory_used": False,
            "selection_policy": args.selection_policy,
            "query_contract": args.query_contract,
        })

    payload = {
        "schema_version": "memp.travelplanner.phase_dynamic_retrieval.v1",
        "split": args.split,
        "task_slice": [min(indices), max(indices) + 1],
        "selected_indices": indices,
        "task_count": len(indices),
        "phase_query_count": len(texts),
        "parent_memory_count": 45,
        "operation_view_count": len(operations),
        "candidate_budget_per_phase": args.candidate_budget or len(operations),
        "per_phase": args.per_phase,
        "final_operation_budget": len(PHASES) * args.per_phase,
        "reranker": args.reranker,
        "score_fusion": False,
        "task_decomposition": "fixed_visible_five_phase_contract_v1",
        "query_contract": args.query_contract,
        "selection_policy": args.selection_policy,
        "official_gold_reward_or_trajectory_used": False,
        "retrievals": retrievals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "retrievals"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
