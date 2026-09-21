#!/usr/bin/env python3
"""Paired retrieval evaluation on the exact first-search queries from Codex runs.

The agent is not called.  We recover each task's first successful
``skilldag graph search`` command, then feed those identical subqueries to:

1. original SkillDAG cosine retrieval; and
2. the V3 hybrid run's actual online search reranker (the same raw subquery,
   alpha=0.25 NCF fusion, and graph diversity).

Gold skill families are the same phase-level semantic definitions used by
``compare_phase_retrieval.py``.  No environment reward or expert trajectory
is used to define relevance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.alfworld.phase_context import build_phase_specs, build_task_structure
from benchmarks.alfworld.phase_online import _cosine, _normalize, _select_diverse
from skilldag.graph import SkillGraph
from skilldag.ncf_reranker import NeuMFReranker

from compare_phase_retrieval import PHASE_GOLD, STATE_GOLD, aggregate, first_rank


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "src/SkillDAG_NCF/results/alfworld"
DEFAULT_MAIN = RESULTS / "skilldag_ncf_v3_hybrid_codex_yunwu_once"
DEFAULT_RETRY = RESULTS / "skilldag_ncf_v3_hybrid_codex_yunwu_retry24_diag"
DEFAULT_GRAPH = ROOT / ".runtime/skilldag/data/skilldag/skilldag_graphs/skillgraph_alfworld.json"
DEFAULT_MODEL = ROOT / ".runtime/skilldag_ncf_v3/models/content_neumf_v1/neumf.pt"
DEFAULT_CALIBRATION = ROOT / ".runtime/skilldag_ncf_v3/models/content_neumf_v1/online_calibration.json"
DEFAULT_CACHE = ROOT / ".runtime/skilldag_ncf_v3/cache/codex_first_search_embeddings.json"
DEFAULT_OUTPUT = ROOT / ".runtime/skilldag_ncf_v3/reports/codex_first_search_paired.json"
DEFAULT_ALFWORLD = ROOT / ".runtime/skilldag/data/alfworld"
RETRY_FILL = {53, 54, 55, 56, 57, 58, 59, 60, 63, 112, 113, 114}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_first_search(result: dict[str, Any]) -> list[str]:
    invocation = next(
        row
        for row in result.get("cli_invocations", [])
        if row.get("rc") == 0 and "skilldag graph search" in row.get("command", "")
    )
    tokens = shlex.split(invocation["command"])
    cursor = tokens.index("search") + 1
    queries: list[str] = []
    while cursor < len(tokens) and not tokens[cursor].startswith("--"):
        queries.append(tokens[cursor].strip())
        cursor += 1
    if not queries:
        raise ValueError("first successful search contains no query")
    return queries


def classify_phase(query: str, structure: Any) -> str:
    """Map a terse agent query to a deterministic ALFWorld phase."""
    text = query.lower().strip()
    first = text.split(maxsplit=1)[0]
    if first in {"clean", "heat", "cool"}:
        return "transform_object"
    if first == "slice":
        return "slice_object"
    if first == "put":
        return "place_objects" if structure.object_count == 2 else "place_object"
    if first in {"look", "examine"}:
        return "assemble"
    if first == "verify":
        return "verify_count" if structure.object_count == 2 else "verify_goal"
    if first == "inventory":
        return "locate_distinct_objects"
    if first == "find":
        if structure.task_type == "look_at_obj_in_light" and structure.device in text:
            return "locate_device"
        return "locate_distinct_objects" if structure.object_count == 2 else "locate_object"
    raise ValueError(f"cannot classify first-search query: {query!r}")


def gold_for(phase: str, required_state: str) -> set[str]:
    if phase == "transform_object":
        return set(STATE_GOLD[required_state])
    return set(PHASE_GOLD[phase])


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_with_cache(graph: SkillGraph, texts: list[str], path: Path) -> dict[str, list[float]]:
    model = os.environ.get("SKILLDAG_EMBEDDING_MODEL", "text-embedding-3-large")
    cache: dict[str, Any] = read_json(path) if path.exists() else {}
    entries = cache.get("entries", {}) if cache.get("model") == model else {}
    unique = list(dict.fromkeys(texts))
    missing = [text for text in unique if text_hash(text) not in entries]
    if missing:
        vectors = graph._embed_queries_batch(missing)
        for text, vector in zip(missing, vectors):
            entries[text_hash(text)] = {"text": text, "embedding": vector}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model": model, "entries": entries}, ensure_ascii=False),
            encoding="utf-8",
        )
    return {text: entries[text_hash(text)]["embedding"] for text in unique}


def load_rows(main: Path, retry: Path, alfworld: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(140):
        source = retry if idx in RETRY_FILL else main
        result = read_json(source / f"idx_{idx}.json")
        trajectory = read_json(
            alfworld / "json_2.1.1" / "valid_seen" / result["name"] / "traj_data.json"
        )
        structure = build_task_structure(trajectory["task_type"], trajectory["pddl_params"])
        phase_queries = {phase.phase_query: phase.phase_name for phase in build_phase_specs(structure)}
        planned_ids = list(
            dict.fromkeys(
                match["skill_id"]
                for phase in result.get("phase_recommendations", [])
                for match in phase.get("matches", [])
            )
        )
        for position, query in enumerate(extract_first_search(result)):
            phase = phase_queries.get(query) or classify_phase(query, structure)
            rows.append(
                {
                    "idx": idx,
                    "task_id": result["name"],
                    "task_type": trajectory["task_type"],
                    "position": position,
                    "query": query,
                    "exact_phase_query": query in phase_queries,
                    "phase": phase,
                    "required_state": structure.required_state,
                    "gold": sorted(gold_for(phase, structure.required_state)),
                    "full_task": result["query"],
                    "planned_ids": planned_ids,
                    "source_result": str(source / f"idx_{idx}.json"),
                }
            )
    return rows


def evaluate(
    rows: list[dict[str, Any]],
    graph: SkillGraph,
    model: Path,
    calibration_path: Path,
    cache: Path,
    top_k: int,
) -> None:
    vectors = embed_with_cache(
        graph, [row["query"] for row in rows], cache
    )
    skill_vectors = graph._load_or_build_node_embeddings()
    skill_ids = sorted(skill_vectors)
    reranker = NeuMFReranker(model, input_dim=len(next(iter(vectors.values()))))
    calibration = read_json(calibration_path)
    alpha = float(calibration["alpha"])
    priors = calibration["skill_logit_prior"]

    for row in rows:
        query_vector = vectors[row["query"]]
        cosine_scores = {
            skill_id: _cosine(query_vector, skill_vectors[skill_id]) for skill_id in skill_ids
        }
        cosine_ranking = sorted(skill_ids, key=lambda sid: (-cosine_scores[sid], sid))

        candidate_ids = cosine_ranking[:12]
        logits = reranker.score(
            query_vector, [skill_vectors[skill_id] for skill_id in candidate_ids]
        )
        cosine_z = _normalize([cosine_scores[skill_id] for skill_id in candidate_ids])
        specific_logits = [
            logit - float(priors.get(skill_id, 0.0))
            for skill_id, logit in zip(candidate_ids, logits)
        ]
        ncf_z = _normalize(specific_logits)
        candidates = [
            {
                "skill_id": skill_id,
                "cosine_score": cosine_scores[skill_id],
                "ncf_logit": logit,
                "base_score": alpha * cz + (1.0 - alpha) * nz,
            }
            for skill_id, logit, cz, nz in zip(candidate_ids, logits, cosine_z, ncf_z)
        ]
        candidates.sort(key=lambda item: (-item["base_score"], item["skill_id"]))
        selected = _select_diverse(candidates, graph, set(cosine_ranking[:3]), top_k=top_k)

        baseline_top = cosine_ranking[:top_k]
        ncf_top = [item["skill_id"] for item in selected]
        gold = set(row["gold"])
        row["skilldag_top_k"] = baseline_top
        row["skilldag_ncf_top_k"] = ncf_top
        row["skilldag_rank"] = first_rank(baseline_top, gold)
        row["skilldag_ncf_rank"] = first_rank(ncf_top, gold)


def summarize(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    return {
        "n_queries": len(rows),
        "SkillDAG": aggregate(rows, "skilldag_rank", top_k),
        "SkillDAG_NCF": aggregate(rows, "skilldag_ncf_rank", top_k),
        "paired": {
            "ncf_better": sum(
                (row["skilldag_ncf_rank"] or top_k + 1) < (row["skilldag_rank"] or top_k + 1)
                for row in rows
            ),
            "same": sum(
                (row["skilldag_ncf_rank"] or top_k + 1) == (row["skilldag_rank"] or top_k + 1)
                for row in rows
            ),
            "skilldag_better": sum(
                (row["skilldag_ncf_rank"] or top_k + 1) > (row["skilldag_rank"] or top_k + 1)
                for row in rows
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-results", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--retry-results", type=Path, default=DEFAULT_RETRY)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alfworld-data", type=Path, default=DEFAULT_ALFWORLD)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows = load_rows(args.main_results, args.retry_results, args.alfworld_data)
    graph = SkillGraph.load(str(args.graph))
    evaluate(rows, graph, args.model, args.calibration, args.cache, args.top_k)
    exact = [row for row in rows if row["exact_phase_query"]]
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_phase[row["phase"]].append(row)
    output = {
        "schema_version": "skilldag_ncf.codex_first_search_paired.v1",
        "definition": {
            "unit": "subquery from the first successful Codex graph-search command",
            "control": "identical task, identical query text, identical frozen skill graph and gold",
            "gold": "fixed semantic phase skill families; no reward/expert trace",
            "ncf_runtime": "same raw subquery + calibrated NCF with skill-prior removal + graph diversity",
            "proactive_plan_note": "phase recommendations were shown to the agent but ALFWorld did not create ncf_plan.json, so online search had no full-task context or plan-safety injection",
            "top_k": args.top_k,
        },
        "overall": summarize(rows, args.top_k),
        "exact_phase_query_only": summarize(exact, args.top_k),
        "by_phase": {
            phase: summarize(phase_rows, args.top_k)
            for phase, phase_rows in sorted(by_phase.items())
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("overall", "exact_phase_query_only")}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
