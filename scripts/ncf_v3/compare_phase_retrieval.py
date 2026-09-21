#!/usr/bin/env python3
"""Compare SkillDAG cosine retrieval with the frozen V3 NCF reranker.

This is an intrinsic, phase-level evaluation.  It never calls a chat model and
never reads evaluation rewards or expert action traces.  Gold sets are fixed
semantic skill families keyed only by the deterministic ALFWorld phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.alfworld.phase_context import (
    build_phase_specs,
    build_task_structure,
    format_phase_context,
)
from benchmarks.alfworld.phase_online import _cosine, _normalize, _select_diverse
from skilldag.graph import SkillGraph
from skilldag.ncf_reranker import NeuMFReranker


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    ROOT
    / "src/SkillDAG_NCF/results/alfworld/pilot10_skilldag_yunwu_once_20260727"
)
DEFAULT_GRAPH = (
    ROOT / ".runtime/skilldag/data/skilldag/skilldag_graphs/skillgraph_alfworld.json"
)
DEFAULT_MODEL = ROOT / ".runtime/skilldag_ncf_v3/models/content_neumf_v1/neumf.pt"
DEFAULT_CALIBRATION = (
    ROOT
    / ".runtime/skilldag_ncf_v3/models/content_neumf_v1/online_calibration.json"
)
DEFAULT_CACHE = (
    ROOT / ".runtime/skilldag_ncf_v3/cache/eval_phase_embeddings_valid_seen.json"
)
DEFAULT_OUTPUT = (
    ROOT / ".runtime/skilldag_ncf_v3/reports/phase_retrieval_comparison.json"
)
DEFAULT_ALFWORLD = ROOT / ".runtime/skilldag/data/alfworld"


PHASE_GOLD: dict[str, set[str]] = {
    "locate_object": {
        "alfworld-object-locator",
        "alfworld-locate-target-object",
        "alfworld-search-pattern-executor",
        "alfworld-tool-locator",
        "alfworld-object-retriever",
    },
    "locate_distinct_objects": {
        "alfworld-object-locator",
        "alfworld-locate-target-object",
        "alfworld-search-pattern-executor",
        "alfworld-object-retriever",
        "alfworld-inventory-management",
    },
    "locate_device": {
        "alfworld-tool-locator",
        "alfworld-appliance-navigator",
        "alfworld-object-locator",
        "alfworld-locate-target-object",
        "alfworld-search-pattern-executor",
    },
    "locate_movable_receptacle": {
        "alfworld-receptacle-finder",
        "alfworld-receptacle-searcher",
        "alfworld-receptacle-navigator",
        "alfworld-storage-explorer",
    },
    "slice_object": {"alfworld-tool-user"},
    "operate_device": {
        "alfworld-device-operator",
        "alfworld-tool-user",
    },
    "assemble": {
        "alfworld-object-placer",
        "alfworld-object-storer",
        "alfworld-object-transporter",
    },
    "place_object": {
        "alfworld-object-placer",
        "alfworld-object-storer",
        "alfworld-object-transporter",
    },
    "place_objects": {
        "alfworld-object-placer",
        "alfworld-object-storer",
        "alfworld-object-transporter",
        "alfworld-inventory-management",
    },
    "place_movable_receptacle": {
        "alfworld-object-placer",
        "alfworld-object-storer",
        "alfworld-object-transporter",
    },
    "verify_goal": {
        "alfworld-task-verifier",
        "alfworld-object-state-inspector",
        "alfworld-search-verifier",
    },
    "verify_count": {
        "alfworld-task-verifier",
        "alfworld-object-state-inspector",
        "alfworld-search-verifier",
        "alfworld-inventory-management",
    },
}

STATE_GOLD = {
    "clean": {
        "alfworld-clean-object",
        "alfworld-object-state-modifier",
        "alfworld-tool-user",
    },
    "heat": {
        "alfworld-object-heater",
        "alfworld-heat-object-with-appliance",
        "alfworld-object-state-modifier",
    },
    "cool": {
        "alfworld-object-cooler",
        "alfworld-object-state-modifier",
        "alfworld-temperature-regulator",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gold_for(phase_name: str, required_state: str) -> set[str]:
    if phase_name == "transform_object":
        return set(STATE_GOLD[required_state])
    return set(PHASE_GOLD[phase_name])


def load_rows(result_dir: Path, alfworld_data: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("idx_*.json")):
        result = read_json(path)
        traj_path = (
            alfworld_data / "json_2.1.1" / "valid_seen" / result["name"] / "traj_data.json"
        )
        trajectory = read_json(traj_path)
        structure = build_task_structure(
            trajectory["task_type"], trajectory["pddl_params"]
        )
        for phase in build_phase_specs(structure):
            context = format_phase_context(
                raw_full_task=result["query"],
                structure=structure,
                phase=phase,
            )
            rows.append(
                {
                    "idx": int(path.stem.split("_")[1]),
                    "task_id": result["name"],
                    "task_type": trajectory["task_type"],
                    "phase": phase.phase_name,
                    "phase_group": phase.phase_group,
                    "query": phase.phase_query,
                    "context": context,
                    "required_state": structure.required_state,
                    "gold": sorted(gold_for(phase.phase_name, structure.required_state)),
                }
            )
    return rows


def embed_with_cache(
    graph: SkillGraph, texts: list[str], cache_path: Path
) -> dict[str, list[float]]:
    model = os.environ.get("SKILLDAG_EMBEDDING_MODEL", "text-embedding-3-large")
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = read_json(cache_path)
    entries = cache.get("entries", {}) if cache.get("model") == model else {}
    missing = [text for text in dict.fromkeys(texts) if text_hash(text) not in entries]
    if missing:
        vectors = graph._embed_queries_batch(missing)
        for text, vector in zip(missing, vectors):
            entries[text_hash(text)] = {"text": text, "embedding": vector}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"model": model, "entries": entries}, ensure_ascii=False),
            encoding="utf-8",
        )
    return {text: entries[text_hash(text)]["embedding"] for text in texts}


def first_rank(ranking: list[str], gold: set[str]) -> int | None:
    return next((rank for rank, skill_id in enumerate(ranking, 1) if skill_id in gold), None)


def aggregate(rows: list[dict[str, Any]], rank_key: str, top_k: int) -> dict[str, Any]:
    ranks = [row[rank_key] for row in rows]
    n = len(ranks)
    return {
        "n_queries": n,
        f"Ret@{top_k}": sum(rank is not None for rank in ranks) / n,
        "Ret@1": sum(rank == 1 for rank in ranks) / n,
        "MRR": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alfworld-data", type=Path, default=DEFAULT_ALFWORLD)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows = load_rows(args.results, args.alfworld_data)
    graph = SkillGraph.load(str(args.graph))
    vectors = embed_with_cache(
        graph,
        [text for row in rows for text in (row["query"], row["context"])],
        args.cache,
    )
    skill_vectors = graph._load_or_build_node_embeddings()
    skill_ids = sorted(skill_vectors)
    reranker = NeuMFReranker(args.model, input_dim=len(next(iter(vectors.values()))))
    calibration = read_json(args.calibration)
    alpha = float(calibration.get("alpha", 0.25))
    priors = calibration.get("skill_logit_prior", {})

    for row in rows:
        query_vector = vectors[row["query"]]
        context_vector = vectors[row["context"]]
        cosine_all = {
            skill_id: _cosine(query_vector, skill_vectors[skill_id])
            for skill_id in skill_ids
        }
        cosine_ranking = sorted(skill_ids, key=lambda sid: (-cosine_all[sid], sid))
        candidate_ids = cosine_ranking[:12]
        logits = reranker.score(
            context_vector, [skill_vectors[skill_id] for skill_id in candidate_ids]
        )
        specific = [
            score - float(priors.get(skill_id, 0.0))
            for skill_id, score in zip(candidate_ids, logits)
        ]
        cosine_z = _normalize([cosine_all[skill_id] for skill_id in candidate_ids])
        ncf_z = _normalize(specific)
        candidates = [
            {
                "skill_id": skill_id,
                "cosine_score": cosine_all[skill_id],
                "ncf_logit": logit,
                "base_score": alpha * cz + (1.0 - alpha) * nz,
            }
            for skill_id, logit, cz, nz in zip(
                candidate_ids, logits, cosine_z, ncf_z
            )
        ]
        candidates.sort(key=lambda item: (-item["base_score"], item["skill_id"]))
        selected = _select_diverse(
            candidates, graph, set(cosine_ranking[:3]), top_k=args.top_k
        )
        cosine_top = cosine_ranking[: args.top_k]
        ncf_top = [item["skill_id"] for item in selected]
        gold = set(row["gold"])
        row["skilldag_top_k"] = cosine_top
        row["skilldag_ncf_top_k"] = ncf_top
        row["skilldag_rank"] = first_rank(cosine_top, gold)
        row["skilldag_ncf_rank"] = first_rank(ncf_top, gold)

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["phase_group"]].append(row)
    output = {
        "schema_version": "skilldag_ncf.phase_retrieval.v1",
        "definition": {
            "unit": "deterministic ALFWorld phase query",
            "gold": "fixed semantic skill families; no rewards or evaluation trajectories",
            "Ret@K": "fraction of phase queries with any gold skill in top K",
            "Ret@1": "fraction with a gold skill ranked first",
            "MRR": "mean reciprocal rank of first gold skill, truncated at K",
            "top_k": args.top_k,
        },
        "inputs": {
            "results": str(args.results),
            "graph": str(args.graph),
            "model": str(args.model),
            "calibration": str(args.calibration),
            "embedding_cache": str(args.cache),
        },
        "overall": {
            "SkillDAG": aggregate(rows, "skilldag_rank", args.top_k),
            "SkillDAG_NCF_V3": aggregate(rows, "skilldag_ncf_rank", args.top_k),
        },
        "by_phase_group": {
            group: {
                "SkillDAG": aggregate(group_rows, "skilldag_rank", args.top_k),
                "SkillDAG_NCF_V3": aggregate(
                    group_rows, "skilldag_ncf_rank", args.top_k
                ),
            }
            for group, group_rows in sorted(by_group.items())
        },
        "paired": {
            "ncf_better_rank": sum(
                (row["skilldag_ncf_rank"] or args.top_k + 1)
                < (row["skilldag_rank"] or args.top_k + 1)
                for row in rows
            ),
            "same_rank": sum(
                (row["skilldag_ncf_rank"] or args.top_k + 1)
                == (row["skilldag_rank"] or args.top_k + 1)
                for row in rows
            ),
            "skilldag_better_rank": sum(
                (row["skilldag_ncf_rank"] or args.top_k + 1)
                > (row["skilldag_rank"] or args.top_k + 1)
                for row in rows
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output["overall"], indent=2, ensure_ascii=False))
    print(json.dumps(output["by_phase_group"], indent=2, ensure_ascii=False))
    print(json.dumps(output["paired"], indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
