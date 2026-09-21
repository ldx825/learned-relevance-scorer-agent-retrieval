#!/usr/bin/env python3
"""Validate SkillDAG graph inputs before launching experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def warn(message: str) -> None:
    print(f"WARN: {message}", file=sys.stderr)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"failed to read JSON {path}: {exc}")


def skill_ids(skills_dir: Path) -> set[str]:
    if not skills_dir.exists():
        fail(f"skills dir missing: {skills_dir}")
    ids: set[str] = set()
    for child in skills_dir.iterdir():
        if child.is_dir() and (child / "SKILL.md").exists():
            ids.add(child.name)
    if not ids:
        fail(f"no SKILL.md directories found directly under {skills_dir}")
    return ids


def cache_path_for(graph_path: Path) -> Path:
    return graph_path.with_suffix(".embeddings.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--skills-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--strict-cache-model", action="store_true")
    parser.add_argument("--allow-extra-skills", action="store_true")
    args = parser.parse_args()

    if not args.graph_path.exists():
        fail(f"graph missing: {args.graph_path}")

    graph = load_json(args.graph_path)
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        fail(f"graph has no nodes: {args.graph_path}")
    graph_ids = set(nodes)
    body_ids = skill_ids(args.skills_dir)

    missing_bodies = sorted(graph_ids - body_ids)
    if missing_bodies:
        sample = ", ".join(missing_bodies[:8])
        fail(
            f"{len(missing_bodies)} graph node(s) have no matching SKILL.md "
            f"under {args.skills_dir}: {sample}"
        )

    extra_bodies = sorted(body_ids - graph_ids)
    if extra_bodies and not args.allow_extra_skills:
        sample = ", ".join(extra_bodies[:8])
        fail(
            f"{len(extra_bodies)} skill dir(s) are not present in graph "
            f"{args.graph_path}: {sample}"
        )

    cache_path = cache_path_for(args.graph_path)
    if not cache_path.exists():
        if args.require_cache:
            fail(f"embedding cache missing: {cache_path}")
        warn(f"embedding cache missing: {cache_path}")
        print(f"OK: graph={len(graph_ids)} skills={len(body_ids)} cache=missing")
        return

    cache = load_json(cache_path)
    if not isinstance(cache, dict):
        fail(f"embedding cache is not an object: {cache_path}")
    missing_cache = sorted(graph_ids - set(cache))
    if missing_cache:
        sample = ", ".join(missing_cache[:8])
        fail(f"{len(missing_cache)} graph node(s) missing embedding cache: {sample}")

    model = args.embedding_model
    cache_models = sorted(
        {
            entry.get("model")
            for entry in cache.values()
            if isinstance(entry, dict) and entry.get("model")
        }
    )
    if model and cache_models and cache_models != [model]:
        message = (
            f"embedding cache model(s) {cache_models} do not match "
            f"SKILLDAG_EMBEDDING_MODEL={model}"
        )
        if args.strict_cache_model:
            fail(message)
        warn(message)

    print(
        f"OK: graph={len(graph_ids)} skills={len(body_ids)} "
        f"cache={len(cache)} model={','.join(cache_models) or 'unknown'}"
    )


if __name__ == "__main__":
    main()
