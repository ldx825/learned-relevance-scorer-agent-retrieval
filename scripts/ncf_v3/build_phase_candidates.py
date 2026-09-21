#!/usr/bin/env python3
"""Build leakage-safe candidate pools for every deterministic ALFWorld phase.

Only phase-context embeddings can trigger API calls. Existing full-task and
skill embeddings are reused, and the frozen V2 NeuMF checkpoint contributes
hard candidates but never contributes labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src/SkillDAG_NCF"
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from skilldag.initialize import _embed_batch  # noqa: E402
from skilldag.ncf_reranker import NeuMFReranker  # noqa: E402


DATA_ROOT = ROOT / "data/alfworld_task_skill"
DEFAULT_PHASE_DIR = DATA_ROOT / "task_skill_v3_phase/phases"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "task_skill_v3_phase/candidates"
DEFAULT_SKILLS = DATA_ROOT / "shared/raw/skills_37.jsonl"
DEFAULT_SKILL_EMBEDDINGS = DATA_ROOT / "shared/embeddings/skill_embeddings_37.json"
DEFAULT_TASK_EMBEDDINGS = DATA_ROOT / "shared/embeddings/task_embeddings_official_train"
DEFAULT_GRAPH = DATA_ROOT / "shared/skillgraph_alfworld.json"
DEFAULT_NCF_MODEL = ROOT / ".runtime/gos_ncf/models/neumf_graded_v2_clean/neumf_graded.pt"
GRAPH_EDGE_TYPES = {"depends_on", "composes_with"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


def top_k(
    query: list[float], skill_matrix: np.ndarray, skill_ids: list[str], count: int
) -> list[tuple[str, float, int]]:
    vector = normalized_matrix([query])[0]
    scores = skill_matrix @ vector
    order = np.argsort(-scores, kind="stable")[:count]
    return [
        (skill_ids[int(index)], float(scores[int(index)]), rank)
        for rank, index in enumerate(order, 1)
    ]


def load_task_embeddings(directory: Path, selected_ids: set[str], model: str) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        for record_id in selected_ids & set(payload):
            entry = payload[record_id]
            if entry.get("model") != model or not entry.get("embedding"):
                raise ValueError(f"invalid full-task embedding cache entry: {record_id}")
            found[record_id] = entry["embedding"]
    missing = selected_ids - set(found)
    if missing:
        raise ValueError(f"missing {len(missing)} cached full-task embeddings")
    return found


def load_or_embed_phases(
    phases: list[dict[str, Any]], cache_dir: Path, model: str, batch_size: int
) -> tuple[dict[str, list[float]], int, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors: dict[str, list[float]] = {}
    reused = 0
    computed = 0
    for batch_index, start in enumerate(range(0, len(phases), batch_size)):
        batch = phases[start : start + batch_size]
        shard_path = cache_dir / f"shard_{batch_index:05d}.json"
        shard = load_json(shard_path) if shard_path.exists() else {}
        missing: list[tuple[dict[str, Any], str]] = []
        for phase in batch:
            phase_id = phase["phase_id"]
            text_hash = sha256_text(phase["context_text"])
            entry = shard.get(phase_id, {})
            if (
                entry.get("model") == model
                and entry.get("text_hash") == text_hash
                and isinstance(entry.get("embedding"), list)
                and entry["embedding"]
            ):
                vectors[phase_id] = entry["embedding"]
                reused += 1
            else:
                missing.append((phase, text_hash))
        if missing:
            generated = _embed_batch([phase["context_text"] for phase, _ in missing])
            if len(generated) != len(missing):
                raise ValueError("embedding API returned the wrong number of vectors")
            for (phase, text_hash), vector in zip(missing, generated, strict=True):
                if not isinstance(vector, list) or not vector:
                    raise ValueError(f"invalid phase embedding: {phase['phase_id']}")
                vectors[phase["phase_id"]] = vector
                shard[phase["phase_id"]] = {
                    "text_hash": text_hash,
                    "model": model,
                    "embedding": vector,
                }
                computed += 1
            atomic_json(shard_path, shard)
        print(
            f"[phase-embedding] batch {batch_index + 1}/"
            f"{(len(phases) + batch_size - 1) // batch_size}: "
            f"items={len(batch)} new_api_inputs={len(missing)}",
            flush=True,
        )
    return vectors, reused, computed


def graph_adjacency(graph: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    adjacency: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in graph.get("edges", []):
        edge_type = str(edge.get("type", ""))
        if edge_type not in GRAPH_EDGE_TYPES:
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            adjacency[source].append(
                {
                    "skill_id": target,
                    "edge_type": edge_type,
                    "edge_source": source,
                    "edge_target": target,
                }
            )
            adjacency[target].append(
                {
                    "skill_id": source,
                    "edge_type": edge_type,
                    "edge_source": source,
                    "edge_target": target,
                }
            )
    return adjacency


def distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    middle = len(ordered) // 2
    median = (
        float(ordered[middle])
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "max": ordered[-1],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    phases = load_jsonl(args.phase_dir / "phases.jsonl")
    tasks = load_jsonl(args.phase_dir / "tasks.jsonl")
    skills = load_jsonl(args.skills_path)
    graph = load_json(args.graph_path)
    skill_cache = load_json(args.skill_embeddings_path)
    phases.sort(key=lambda row: row["phase_id"])
    skill_ids = sorted(row["skill_id"] for row in skills)
    if set(skill_ids) != set(skill_cache) or set(skill_ids) != set(graph["nodes"]):
        raise ValueError("skills, skill embeddings, and graph nodes do not match")
    if any(skill_cache[skill_id].get("model") != args.embedding_model for skill_id in skill_ids):
        raise ValueError("skill embedding model mismatch")

    skill_vectors = [skill_cache[skill_id]["embedding"] for skill_id in skill_ids]
    skill_matrix = normalized_matrix(skill_vectors)
    selected_task_ids = {task["task_record_id"] for task in tasks}
    full_task_vectors = load_task_embeddings(
        args.task_embeddings_dir, selected_task_ids, args.embedding_model
    )
    phase_vectors, reused, computed = load_or_embed_phases(
        phases,
        args.output_dir / "embedding_cache" / args.embedding_model,
        args.embedding_model,
        args.batch_size,
    )
    expected_dim = len(skill_vectors[0])
    if any(len(vector) != expected_dim for vector in phase_vectors.values()):
        raise ValueError("phase/skill embedding dimension mismatch")

    reranker = NeuMFReranker(args.ncf_model, input_dim=expected_dim)
    adjacency = graph_adjacency(graph)
    candidate_rows: list[dict[str, Any]] = []
    candidate_counts: list[int] = []
    source_counts: Counter[str] = Counter()
    skill_exposure: Counter[str] = Counter()
    near_full_library = 0

    for phase in phases:
        phase_id = phase["phase_id"]
        phase_vector = phase_vectors[phase_id]
        merged: dict[str, dict[str, Any]] = {}

        def add_ranked(source: str, ranked: list[tuple[str, float, int]]) -> None:
            for skill_id, score, rank in ranked:
                row = merged.setdefault(
                    skill_id,
                    {"skill_id": skill_id, "candidate_sources": [], "graph_paths": []},
                )
                row["candidate_sources"].append(source)
                row[f"{source}_score"] = score
                row[f"{source}_rank"] = rank

        add_ranked(
            "phase_cosine",
            top_k(phase_vector, skill_matrix, skill_ids, args.phase_top_k),
        )
        add_ranked(
            "full_task_cosine",
            top_k(
                full_task_vectors[phase["task_record_id"]],
                skill_matrix,
                skill_ids,
                args.full_task_top_k,
            ),
        )
        ncf_scores = reranker.score(phase_vector, skill_vectors)
        ncf_ranked = sorted(
            zip(skill_ids, ncf_scores), key=lambda item: (-item[1], item[0])
        )[: args.ncf_top_k]
        add_ranked(
            "v2_ncf_hard",
            [(skill_id, float(score), rank) for rank, (skill_id, score) in enumerate(ncf_ranked, 1)],
        )

        seed_ids = sorted(merged)
        for seed_id in seed_ids:
            for neighbor in adjacency.get(seed_id, []):
                skill_id = neighbor["skill_id"]
                row = merged.setdefault(
                    skill_id,
                    {"skill_id": skill_id, "candidate_sources": [], "graph_paths": []},
                )
                if "graph_one_hop" not in row["candidate_sources"]:
                    row["candidate_sources"].append("graph_one_hop")
                path = {key: value for key, value in neighbor.items() if key != "skill_id"}
                path["reached_from"] = seed_id
                if path not in row["graph_paths"]:
                    row["graph_paths"].append(path)

        candidate_counts.append(len(merged))
        near_full_library += int(len(merged) >= len(skill_ids) - 2)
        for skill_id, evidence in sorted(merged.items()):
            skill_exposure[skill_id] += 1
            for source in evidence["candidate_sources"]:
                source_counts[source] += 1
            candidate_rows.append(
                {
                    "schema_version": "skilldag_ncf.v3.phase_candidate.v1",
                    "phase_id": phase_id,
                    "task_record_id": phase["task_record_id"],
                    "internal_split": phase["internal_split"],
                    "task_type": phase["task_type"],
                    "phase_index": phase["phase_index"],
                    "phase_name": phase["phase_name"],
                    "phase_group": phase["phase_group"],
                    "skill_id": skill_id,
                    **evidence,
                    "candidate_only": True,
                    "label": None,
                }
            )

    report = {
        "schema_version": "skilldag_ncf.v3.phase_candidates_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "phase_top_k": args.phase_top_k,
            "full_task_top_k": args.full_task_top_k,
            "v2_ncf_top_k": args.ncf_top_k,
            "graph_edge_types": sorted(GRAPH_EDGE_TYPES),
            "embedding_model": args.embedding_model,
            "ncf_model": str(args.ncf_model),
        },
        "task_count": len(tasks),
        "phase_count": len(phases),
        "candidate_pair_count": len(candidate_rows),
        "candidates_per_phase": distribution(candidate_counts),
        "near_full_library_phase_count": near_full_library,
        "source_pair_counts": dict(sorted(source_counts.items())),
        "skill_exposure_counts": dict(sorted(skill_exposure.items())),
        "embedding_usage": {
            "phase_vectors_cached_total": len(phase_vectors),
            "phase_vectors_reused_this_run": reused,
            "phase_vectors_computed_this_run": computed,
            "embedding_api_inputs_this_run": computed,
            "full_task_vectors_reused": len(full_task_vectors),
            "skill_vectors_reused": len(skill_vectors),
        },
        "uses_eval_tasks": False,
        "uses_expert_plan": False,
        "labels_generated": 0,
        "outputs": {"candidates": "phase_candidates.jsonl", "report": "report.json"},
    }
    atomic_jsonl(args.output_dir / "phase_candidates.jsonl", candidate_rows)
    atomic_json(args.output_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-dir", type=Path, default=DEFAULT_PHASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skills-path", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument("--skill-embeddings-path", type=Path, default=DEFAULT_SKILL_EMBEDDINGS)
    parser.add_argument("--task-embeddings-dir", type=Path, default=DEFAULT_TASK_EMBEDDINGS)
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--ncf-model", type=Path, default=DEFAULT_NCF_MODEL)
    parser.add_argument("--embedding-model", default=os.environ.get("SKILLDAG_EMBEDDING_MODEL", "text-embedding-3-large"))
    parser.add_argument("--phase-top-k", type=int, default=12)
    parser.add_argument("--full-task-top-k", type=int, default=8)
    parser.add_argument("--ncf-top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for value in (args.phase_top_k, args.full_task_top_k, args.ncf_top_k, args.batch_size):
        if value < 1:
            raise SystemExit("all top-k and batch-size values must be positive")
    if not os.environ.get("SKILLDAG_EMBEDDING_API_KEY"):
        raise SystemExit("SKILLDAG_EMBEDDING_API_KEY is empty; use build_phase_candidates.sh")
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
