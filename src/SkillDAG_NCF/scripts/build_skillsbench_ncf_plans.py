#!/usr/bin/env python3
"""Build leakage-safe proactive NCF plans from SkillsBench instructions.

This is an inference adapter for the original SkillDAG SkillsBench runner.  It
reads only the task-visible ``instruction.md`` plus the frozen skill graph,
embeddings, and NCF bundle.  It never reads task skills, solutions, tests,
verifier files, rewards, trajectories, or prior run outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from skilldag.hybrid_selection import cosine, normalize_scores, select_diverse
from skilldag.initialize import _embed_batch
from skilldag.portable_ncf import PortableNeuMFReranker


SCHEMA_VERSION = "skilldag_ncf.skillsbench_plan.v1"
PROTOCOL_RE = re.compile(
    r"<!-- BEGIN SKILLDAG ONLINE PROTOCOL -->.*?"
    r"<!-- END SKILLDAG ONLINE PROTOCOL -->",
    re.DOTALL,
)
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact(text: str, limit: int = 6000) -> str:
    return " ".join(text.split())[:limit]


def build_task_views(instruction: str, *, max_stage_views: int = 3) -> list[dict[str, str]]:
    """Extract deterministic, task-visible planning views without another LLM."""
    clean = PROTOCOL_RE.sub("", instruction).strip()
    full_task = _compact(clean)
    if not full_task:
        raise ValueError("instruction.md is empty")
    views = [{"name": "full_task", "query": full_task}]

    candidates: list[str] = []
    for line in clean.splitlines():
        match = BULLET_RE.match(line)
        if match:
            candidate = _compact(match.group(1), 500)
            if 12 <= len(candidate) <= 500:
                candidates.append(candidate)
    if not candidates:
        candidates = [
            _compact(sentence, 500)
            for sentence in re.split(r"(?<=[.!?])\s+", full_task)
            if 20 <= len(_compact(sentence, 500)) <= 500
        ]

    seen = {full_task.lower()}
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        views.append(
            {
                "name": f"stage_{len(views)}",
                "query": candidate,
            }
        )
        if len(views) > max_stage_views:
            break
    return views


def _load_skill_vectors(path: Path) -> tuple[dict[str, list[float]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", payload)
    if not isinstance(items, dict):
        raise ValueError("skill embedding cache must be a skill-id map or contain an items map")
    vectors = {
        str(skill_id): [float(value) for value in entry["embedding"]]
        for skill_id, entry in items.items()
    }
    models = {str(payload.get("model", ""))}
    models.update(str(entry.get("model", "")) for entry in items.values())
    models.discard("")
    if len(models) > 1:
        raise ValueError(f"embedding cache contains multiple models: {sorted(models)}")
    return vectors, next(iter(models), "")


def _load_cache(path: Path, model: str) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("embedding_model") != model:
        return {}
    return {
        str(key): [float(value) for value in vector]
        for key, vector in (payload.get("vectors") or {}).items()
    }


def _embed_cached(
    texts: list[str],
    *,
    cache_path: Path,
    embedding_model: str,
    batch_size: int,
) -> list[list[float]]:
    cache = _load_cache(cache_path, embedding_model)
    keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    missing = [(key, text) for key, text in zip(keys, texts) if key not in cache]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = _embed_batch([text for _, text in batch])
        for (key, _), vector in zip(batch, vectors):
            cache[key] = [float(value) for value in vector]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"embedding_model": embedding_model, "vectors": cache},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return [cache[key] for key in keys]


def build_manifest(
    *,
    tasks_root: Path,
    graph_path: Path,
    embeddings_path: Path,
    bundle_path: Path,
    output_path: Path,
    embedding_cache_path: Path,
    candidate_k: int = 100,
    top_k: int = 3,
    task_names: list[str] | None = None,
) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes") or {}
    skill_vectors, cache_model = _load_skill_vectors(embeddings_path)
    if set(nodes) != set(skill_vectors):
        raise ValueError("graph nodes and skill embedding cache do not match")
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle_model = str(bundle_payload.get("embedding_model", ""))
    if cache_model and bundle_model and cache_model != bundle_model:
        raise ValueError(
            f"bundle uses {bundle_model!r}, skill cache uses {cache_model!r}"
        )
    embedding_model = bundle_model or cache_model
    configured_model = os.environ.get("SKILLDAG_EMBEDDING_MODEL", embedding_model)
    if configured_model != embedding_model:
        raise ValueError(
            f"SKILLDAG_EMBEDDING_MODEL={configured_model!r}, expected {embedding_model!r}"
        )

    selected = set(task_names or [])
    task_dirs = sorted(path for path in tasks_root.iterdir() if path.is_dir())
    if selected:
        task_dirs = [path for path in task_dirs if path.name in selected]
        missing = sorted(selected - {path.name for path in task_dirs})
        if missing:
            raise FileNotFoundError(f"task directories not found: {missing}")
    if not task_dirs:
        raise ValueError(f"no task directories under {tasks_root}")

    raw_tasks: dict[str, dict[str, Any]] = {}
    all_embedding_texts: list[str] = []
    embedding_slots: list[tuple[str, int, str]] = []
    for task_dir in task_dirs:
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.is_file():
            raise FileNotFoundError(f"missing {instruction_path}")
        instruction = instruction_path.read_text(encoding="utf-8")
        views = build_task_views(instruction)
        raw_tasks[task_dir.name] = {"full_task": views[0]["query"], "views": views}
        for index, view in enumerate(views):
            # Local query drives conservative cosine recall.
            all_embedding_texts.append(view["query"])
            embedding_slots.append((task_dir.name, index, "query"))
            # Full task + current phase matches the V3 train/runtime convention.
            context = (
                f"Full task:\n{views[0]['query']}\n\n"
                f"Current phase:\n{view['query']}"
            )
            all_embedding_texts.append(context)
            embedding_slots.append((task_dir.name, index, "context"))

    vectors = _embed_cached(
        all_embedding_texts,
        cache_path=embedding_cache_path,
        embedding_model=embedding_model,
        batch_size=64,
    )
    embedded: dict[tuple[str, int, str], list[float]] = {
        slot: vector for slot, vector in zip(embedding_slots, vectors)
    }
    reranker = PortableNeuMFReranker(
        bundle_path, input_dim=len(next(iter(skill_vectors.values())))
    )
    calibration = bundle_payload.get("calibration") or {}
    alpha = float(calibration.get("alpha", 0.25))
    priors = calibration.get("skill_logit_prior") or {}
    skill_ids = sorted(skill_vectors)

    tasks: dict[str, Any] = {}
    for task_id, task in raw_tasks.items():
        planned_views = []
        for index, view in enumerate(task["views"]):
            query_vector = embedded[(task_id, index, "query")]
            context_vector = embedded[(task_id, index, "context")]
            cosine_all = {
                skill_id: cosine(query_vector, skill_vectors[skill_id])
                for skill_id in skill_ids
            }
            candidates = sorted(
                skill_ids, key=lambda skill_id: (-cosine_all[skill_id], skill_id)
            )[:candidate_k]
            logits = reranker.score_by_id(context_vector, candidates)
            specific = [
                logit - float(priors.get(skill_id, 0.0))
                for skill_id, logit in zip(candidates, logits)
            ]
            cosine_z = normalize_scores([cosine_all[skill_id] for skill_id in candidates])
            ncf_z = normalize_scores(specific)
            rows = []
            for skill_id, logit, cz, nz in zip(candidates, logits, cosine_z, ncf_z):
                rows.append(
                    {
                        "skill_id": skill_id,
                        "description": str(nodes[skill_id].get("description", "")),
                        "cosine_score": cosine_all[skill_id],
                        "ncf_logit": logit,
                        "base_score": alpha * cz + (1.0 - alpha) * nz,
                    }
                )
            rows.sort(key=lambda row: (-row["base_score"], row["skill_id"]))
            cosine_safe = set(
                sorted(candidates, key=lambda sid: (-cosine_all[sid], sid))[:3]
            )
            chosen = select_diverse(
                rows,
                edges=graph.get("edges", []),
                cosine_safe_ids=cosine_safe,
                top_k=top_k,
            )
            planned_views.append(
                {
                    "name": view["name"],
                    "query": view["query"],
                    "matches": [
                        {
                            "skill_id": row["skill_id"],
                            "description": row["description"],
                        }
                        for row in chosen
                    ],
                }
            )
        tasks[task_id] = {"views": planned_views}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tasks": tasks,
        "generation": {
            "task_count": len(tasks),
            "candidate_k": candidate_k,
            "top_k": top_k,
            "alpha": alpha,
            "official_gold_used_by_planner": False,
            "task_visible_instruction_only": True,
            "graph_sha256": _sha256(graph_path),
            "embeddings_sha256": _sha256(embeddings_path),
            "bundle_sha256": _sha256(bundle_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--task", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest(
        tasks_root=args.tasks_root.resolve(),
        graph_path=args.graph.resolve(),
        embeddings_path=args.embeddings.resolve(),
        bundle_path=args.bundle.resolve(),
        output_path=args.output.resolve(),
        embedding_cache_path=args.embedding_cache.resolve(),
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        task_names=args.task,
    )
    print(f"built proactive plans for {len(payload['tasks'])} tasks → {args.output}")


if __name__ == "__main__":
    main()
