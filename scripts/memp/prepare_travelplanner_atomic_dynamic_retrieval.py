#!/usr/bin/env python3
"""Build decomposed TravelPlanner dynamic-memory retrieval without gold use."""

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
from travelplanner_atomic_memory import decompose_task, render_dynamic_memory  # noqa: E402
from gos.ncf.models import NCFConfig, NeuMF  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def signature(row: dict[str, str]) -> dict[str, Any]:
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


def conflicts(subgoal: dict[str, Any], operation: dict[str, Any]) -> bool:
    required = subgoal.get("required_value")
    offered = operation.get("applicability_value")
    if not required or not offered:
        return False
    left = required if isinstance(required, list) else [required]
    right = offered if isinstance(offered, list) else [offered]
    return {str(value).strip().lower() for value in left} != {
        str(value).strip().lower() for value in right
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--reranker", choices=("neumf", "cosine"), default="neumf")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indices = list(range(args.start, min(args.start + args.count, len(rows))))
    operations = read_jsonl(args.operations)
    if not indices or len(operations) != 519:
        raise ValueError("empty task slice or unexpected operation bank")
    if any(row.get("validation_or_test_used") is not False for row in operations):
        raise ValueError("operation bank is not train-only")

    task_subgoals: list[tuple[int, str, list[dict[str, Any]]]] = []
    for index in indices:
        query = rows[index]["query"]
        decomposed = decompose_task(
            query,
            signature(rows[index]),
            family_id=f"travelplanner_{args.split}_{index:04d}",
            model_split=f"official_{args.split}_inference_only",
        )
        task_subgoals.append((index, query, decomposed))
    query_texts = [subgoal["subgoal_query"] for _, _, values in task_subgoals for subgoal in values]
    if args.dry_run:
        print(json.dumps({
            "task_count": len(indices),
            "subgoal_embedding_count": len(query_texts),
            "operation_count": len(operations),
            "candidate_budget": args.candidate_budget,
            "gold_or_reward_used": False,
        }, indent=2))
        return 0

    vectors = embed_texts(
        query_texts,
        model=args.embedding_model,
        cache_path=args.cache,
        api_base=os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1"),
        api_key=os.environ.get("SKILLDAG_EMBEDDING_API_KEY", ""),
        batch_size=64,
        timeout=180,
    )
    with np.load(args.arrays, allow_pickle=False) as payload:
        operation_embeddings = payload["memory_embeddings"].copy().astype(np.float32)
        operation_ids = [str(value) for value in payload["operation_ids"]]
    if operation_ids != [str(row["operation_id"]) for row in operations]:
        raise ValueError("operation embedding ordering mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NeuMF(NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    op_norm = operation_embeddings / np.maximum(np.linalg.norm(operation_embeddings, axis=1, keepdims=True), 1e-12)

    retrievals, cursor = [], 0
    with torch.no_grad():
        for test_index, query, subgoals in task_subgoals:
            selected = []
            diagnostics = []
            for subgoal in subgoals:
                vector = vectors[cursor].astype(np.float32); cursor += 1
                cosine = op_norm @ (vector / max(float(np.linalg.norm(vector)), 1e-12))
                allowed = np.asarray([
                    index for index, operation in enumerate(operations)
                    if operation["operation_key"] == subgoal["operation_key"]
                    and not conflicts(subgoal, operation)
                ], dtype=np.int32)
                if not len(allowed):
                    raise ValueError(
                        f"no compatible operation for {subgoal['subgoal_id']} "
                        f"value={subgoal.get('required_value')!r}"
                    )
                order = allowed[np.argsort(-cosine[allowed], kind="stable")]
                candidates = order[: args.candidate_budget]
                tasks = torch.from_numpy(np.repeat(vector[None, :], len(candidates), axis=0))
                logits = model(tasks, torch.from_numpy(operation_embeddings[candidates])).numpy()
                best_position = int(np.argmax(logits)) if args.reranker == "neumf" else 0
                best_id = int(candidates[best_position])
                picked = dict(operations[best_id], score=float(logits[best_position]))
                selected.append(picked)
                diagnostics.append({
                    "subgoal_id": subgoal["subgoal_id"],
                    "operation_key": subgoal["operation_key"],
                    "required_value": subgoal.get("required_value"),
                    "selected_operation_id": picked["operation_id"],
                    "selected_operation_key": picked["operation_key"],
                    "parent_memory_id": picked["parent_memory_id"],
                    "cosine_rank_within_allowed": int(np.flatnonzero(order == best_id)[0]) + 1,
                    "ncf_score": float(logits[best_position]),
                })
            retrievals.append({
                "test_index": test_index,
                "query": query,
                "dynamic_memory_text": render_dynamic_memory(query, selected),
                "selected_operations": diagnostics,
                "retriever": (
                    "typed_operation_cosine_top20_then_direct_content_neumf"
                    if args.reranker == "neumf"
                    else "typed_operation_cosine"
                ),
                "gold_or_reward_used": False,
            })
    payload = {
        "schema_version": "memp.travelplanner.atomic_dynamic_retrieval.v1",
        "split": args.split,
        "test_slice": [indices[0], indices[-1] + 1],
        "task_count": len(indices),
        "subgoal_count": len(query_texts),
        "operation_count": len(operations),
        "candidate_budget": args.candidate_budget,
        "embedding_model": args.embedding_model,
        "retriever": (
            "typed_operation_cosine_top20_then_direct_content_neumf"
            if args.reranker == "neumf"
            else "typed_operation_cosine"
        ),
        "score_fusion": False,
        "reranker": args.reranker,
        "official_gold_or_reward_used": False,
        "retrievals": retrievals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "retrievals"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
