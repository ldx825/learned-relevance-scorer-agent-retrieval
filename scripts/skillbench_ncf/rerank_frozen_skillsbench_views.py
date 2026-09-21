#!/usr/bin/env python3
"""Rerank an existing frozen SkillsBench view manifest without API calls.

The input plan supplies task-visible query strings only. Query/context vectors
must already exist in the supplied cache. This makes cosine and NCF comparisons
use byte-identical queries and refuses to spend embedding API credits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from skilldag.hybrid_selection import cosine, normalize_scores, select_diverse
from skilldag.portable_ncf import PortableNeuMFReranker


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_for(cache: dict, text: str) -> list[float]:
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        return [float(value) for value in cache["vectors"][key]]
    except KeyError as error:
        raise KeyError(
            f"frozen embedding missing for sha256={key}; refusing API fallback"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--skill-embeddings", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    source = load_json(args.views)
    cache = load_json(args.query_cache)
    graph = load_json(args.graph)
    embedding_payload = load_json(args.skill_embeddings)
    skill_items = embedding_payload.get("items", embedding_payload)
    bundle_payload = load_json(args.bundle)
    nodes = graph.get("nodes") or {}

    if set(nodes) != set(skill_items):
        raise ValueError("graph nodes and skill embeddings do not match")
    cache_model = str(cache.get("embedding_model", ""))
    bundle_model = str(bundle_payload.get("embedding_model", ""))
    if cache_model and bundle_model and cache_model != bundle_model:
        raise ValueError(f"embedding model mismatch: {cache_model} != {bundle_model}")

    skill_vectors = {
        skill_id: [float(value) for value in entry["embedding"]]
        for skill_id, entry in skill_items.items()
    }
    reranker = PortableNeuMFReranker(
        args.bundle, input_dim=len(next(iter(skill_vectors.values())))
    )
    calibration = bundle_payload.get("calibration") or {}
    alpha = float(calibration.get("alpha", 0.25))
    priors = calibration.get("skill_logit_prior") or {}
    skill_ids = sorted(skill_vectors)

    tasks = {}
    for task_id, task in source["tasks"].items():
        source_views = task["views"]
        full_task = source_views[0]["query"]
        planned = []
        for view in source_views:
            query = view["query"]
            query_vector = vector_for(cache, query)
            context = f"Full task:\n{full_task}\n\nCurrent phase:\n{query}"
            context_vector = vector_for(cache, context)
            cosine_all = {
                skill_id: cosine(query_vector, skill_vectors[skill_id])
                for skill_id in skill_ids
            }
            candidates = sorted(
                skill_ids, key=lambda skill_id: (-cosine_all[skill_id], skill_id)
            )[: args.candidate_k]
            logits = reranker.score_by_id(context_vector, candidates)
            specific = [
                logit - float(priors.get(skill_id, 0.0))
                for skill_id, logit in zip(candidates, logits)
            ]
            cosine_z = normalize_scores([cosine_all[sid] for sid in candidates])
            ncf_z = normalize_scores(specific)
            rows = [
                {
                    "skill_id": skill_id,
                    "description": str(nodes[skill_id].get("description", "")),
                    "cosine_score": cosine_all[skill_id],
                    "ncf_logit": logit,
                    "base_score": alpha * cz + (1.0 - alpha) * nz,
                }
                for skill_id, logit, cz, nz in zip(
                    candidates, logits, cosine_z, ncf_z
                )
            ]
            rows.sort(key=lambda row: (-row["base_score"], row["skill_id"]))
            cosine_safe = set(
                sorted(candidates, key=lambda sid: (-cosine_all[sid], sid))[:3]
            )
            chosen = select_diverse(
                rows,
                edges=graph.get("edges", []),
                cosine_safe_ids=cosine_safe,
                top_k=args.top_k,
            )
            planned.append(
                {
                    "name": view["name"],
                    "query": query,
                    "matches": [
                        {
                            "skill_id": row["skill_id"],
                            "description": row["description"],
                        }
                        for row in chosen
                    ],
                }
            )
        tasks[task_id] = {"views": planned}

    payload = {
        "schema_version": "skilldag_ncf.skillsbench_plan.v1",
        "tasks": tasks,
        "generation": {
            "task_count": len(tasks),
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "alpha": alpha,
            "official_gold_used_by_planner": False,
            "task_visible_instruction_only": True,
            "api_calls": 0,
            "frozen_views_sha256": sha256(args.views),
            "query_cache_sha256": sha256(args.query_cache),
            "graph_sha256": sha256(args.graph),
            "embeddings_sha256": sha256(args.skill_embeddings),
            "bundle_sha256": sha256(args.bundle),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "views": sum(len(task["views"]) for task in tasks.values()),
                "alpha": alpha,
                "api_calls": 0,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
