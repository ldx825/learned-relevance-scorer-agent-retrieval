#!/usr/bin/env python3
"""Offline query replay for cross-scale and cold-vs-edited retrieval metrics.

Usage:
  # Build query manifest from a completed SkillsBench run:
  python scripts/replay_queries.py --mode build-manifest \
    --trials-dir results/skillsbench/<exp>/ \
    --gold-root results/skillsbench_tasks/tasks_skilldag_full_<scale>/ \
    --output analysis/queries.json

  # Cross-scale retrieval: rerun all queries against each scale's graph
  python scripts/replay_queries.py --mode cross-scale \
    --scales 200 500 1000 2000 \
    --graph-dir data/skilldag_graphs \
    --query-manifest analysis/queries.json \
    --output analysis/cross_scale_scores.json

  # Cold-vs-edited: compare two graphs on the same query set
  python scripts/replay_queries.py --mode edit-effect \
    --cold-graph data/skilldag_graphs/skillgraph_1000.json \
    --edited-graph results/alfworld/traintest_<ts>/graph_after_train.json \
    --query-manifest analysis/queries.json \
    --output analysis/edit_effect_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# SkillDAG library
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from skilldag.graph import SkillGraph


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_objects(text: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def extract_search_queries(trial_dir: Path) -> list[str]:
    """Extract search query strings from a SkillsBench trial directory.

    Parses trajectory.json or agent/prompt.txt files to find the queries
    the agent actually searched with.
    """
    queries: list[str] = []

    # Try trajectory.json first
    traj_path = trial_dir / "agent" / "trajectory.json"
    if traj_path.exists():
        try:
            data = load_json(traj_path)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    prompt = item.get("prompt")
                    if not isinstance(prompt, str):
                        continue
                    for obj in json_objects(prompt):
                        if obj.get("ok") is True and obj.get("op") == "search":
                            q = obj.get("query")
                            if isinstance(q, str) and q:
                                queries.append(q)
        except (json.JSONDecodeError, OSError):
            pass

    # Also scan episode prompt/response files for search commands
    agent_dir = trial_dir / "agent"
    if agent_dir.exists():
        for prompt_file in sorted(agent_dir.glob("episode-*/prompt.txt")):
            try:
                text = prompt_file.read_text(encoding="utf-8")
                for obj in json_objects(text):
                    if obj.get("op") == "search":
                        q = obj.get("query")
                        if isinstance(q, str) and q:
                            queries.append(q)
            except OSError:
                continue

    return list(dict.fromkeys(queries))  # dedupe, preserve order


def load_gold_skills(trial_dir: Path, gold_root: Path | None = None) -> list[str]:
    """Load ground-truth skill IDs for a trial from the task's gold_skills.json.

    Args:
        trial_dir: directory of a single trial run.
        gold_root: root directory containing the task directories with gold_skills.json.
                   If None, falls back to the old relative path heuristic.
    """
    task_name = trial_dir.name
    if "__" in task_name:
        task_name = task_name.rsplit("__", 1)[0]

    if gold_root is not None:
        gold_path = gold_root / task_name / "environment" / "skilldag" / "gold_skills.json"
        if gold_path.exists():
            try:
                data = load_json(gold_path)
                return [str(s) for s in data.get("gold_skills", [])]
            except (json.JSONDecodeError, OSError):
                pass
        return []

    # Fallback heuristic for backwards compatibility
    gold_path = trial_dir.parent.parent / "tasks_skilldag_full_" / task_name / "environment" / "skilldag" / "gold_skills.json"
    if gold_path.exists():
        try:
            data = load_json(gold_path)
            return [str(s) for s in data.get("gold_skills", [])]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def build_manifest(trials_dir: Path, output: Path, gold_root: Path | None = None) -> dict[str, Any]:
    """Scan a completed run directory and produce a query manifest.

    The manifest maps task_id -> list of (query_string, gold_skill_ids).
    """
    manifest: dict[str, list[dict[str, Any]]] = {}

    for trial in sorted(trials_dir.iterdir()):
        if not trial.is_dir():
            continue
        task_name = trial.name
        if "__" in task_name:
            task_name = task_name.rsplit("__", 1)[0]

        queries = extract_search_queries(trial)
        gold = load_gold_skills(trial, gold_root=gold_root)

        if task_name not in manifest:
            manifest[task_name] = []
        manifest[task_name].append({"queries": queries, "gold_skills": gold})

    total_gold = sum(len(g) for entries in manifest.values() for g in entries if g)
    if total_gold == 0:
        print("ERROR: no gold_skills found in any trial — check --gold-root path", file=sys.stderr)
        print("       The tasks directory should contain task/environment/skilldag/gold_skills.json files.", file=sys.stderr)
        sys.exit(1)

    result = {
        "mode": "query_manifest",
        "trials_dir": str(trials_dir),
        "gold_root": str(gold_root) if gold_root else None,
        "num_tasks": len(manifest),
        "tasks": manifest,
    }
    save_json(output, result)
    print(f"Wrote query manifest ({len(manifest)} tasks, {total_gold} gold entries) → {output}")
    return result


def retrieval_score(graph: SkillGraph, query: str, gold: list[str], top_k: int = 5) -> dict[str, Any]:
    """Run a single query against a graph and score it."""
    try:
        result = graph.search(query, top_k=top_k)
    except Exception as e:
        return {"query": query, "error": str(e), "hit_at_5": False, "hit_at_1": False, "mrr": 0.0, "rank": None}

    matches = result.get("matches", [])
    retrieved = [m.get("skill_id") for m in matches if m.get("skill_id")]

    first_rank = None
    for i, sid in enumerate(retrieved, 1):
        if sid in gold:
            first_rank = i
            break

    return {
        "query": query,
        "retrieved": retrieved[:top_k],
        "gold": gold,
        "hit_at_5": first_rank is not None,
        "hit_at_1": first_rank == 1,
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "rank": first_rank,
    }


def cross_scale(scales: list[int], graph_dir: Path, query_manifest: Path, output: Path) -> dict[str, Any]:
    """Rerun queries on each scale graph and compute Ret@K / MRR."""
    manifest = load_json(query_manifest)

    all_rows: dict[str, list[dict[str, Any]]] = {str(s): [] for s in scales}

    for scale in scales:
        graph_path = graph_dir / f"skillgraph_{scale}.json"
        if not graph_path.exists():
            print(f"WARNING: graph not found: {graph_path}", file=sys.stderr)
            continue

        graph = SkillGraph.load(str(graph_path))
        print(f"  [{scale}] {len(graph.nodes)} skills, {len(graph.edges)} edges")

        for task_id, entries in manifest.get("tasks", {}).items():
            for entry in entries:
                gold = entry.get("gold_skills", [])
                if not gold:
                    continue
                for q in entry.get("queries", []):
                    score = retrieval_score(graph, q, gold)
                    score["task_id"] = task_id
                    all_rows[str(scale)].append(score)

    # Aggregate per scale
    summary: dict[str, Any] = {"scales": {}}
    for scale_str, rows in all_rows.items():
        if not rows:
            summary["scales"][scale_str] = {"n_queries": 0}
            continue
        n = len(rows)
        hit5 = sum(r["hit_at_5"] for r in rows)
        hit1 = sum(r["hit_at_1"] for r in rows)
        mrr_sum = sum(r["mrr"] for r in rows)
        summary["scales"][scale_str] = {
            "n_queries": n,
            "Ret@5": hit5 / n,
            "Ret@1": hit1 / n,
            "MRR": mrr_sum / n,
        }

    save_json(output, summary)
    print(f"Wrote cross-scale scores → {output}")
    return summary


def edit_effect(cold_graph_path: Path, edited_graph_path: Path, query_manifest: Path, output: Path) -> dict[str, Any]:
    """Compare cold vs edited graph on the same query set."""
    cold_graph = SkillGraph.load(str(cold_graph_path))
    edited_graph = SkillGraph.load(str(edited_graph_path))

    manifest = load_json(query_manifest)

    cold_rows: list[dict[str, Any]] = []
    edited_rows: list[dict[str, Any]] = []

    for task_id, entries in manifest.get("tasks", {}).items():
        for entry in entries:
            gold = entry.get("gold_skills", [])
            if not gold:
                continue
            for q in entry.get("queries", []):
                cold_rows.append(retrieval_score(cold_graph, q, gold))
                edited_rows.append(retrieval_score(edited_graph, q, gold))

    def mean_rank(rows: list[dict[str, Any]]) -> float:
        ranks = [r["rank"] for r in rows if r["rank"] is not None]
        return sum(ranks) / len(ranks) if ranks else 0.0

    summary = {
        "cold_graph": str(cold_graph_path),
        "edited_graph": str(edited_graph_path),
        "cold_mean_rank": mean_rank(cold_rows),
        "edited_mean_rank": mean_rank(edited_rows),
        "n_queries": len(cold_rows),
    }

    save_json(output, summary)
    print(f"Wrote edit-effect metrics → {output}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    m_build = sub.add_parser("build-manifest", help="Build query manifest from trial logs")
    m_build.add_argument("--trials-dir", type=Path, required=True, help="SkillsBench run directory")
    m_build.add_argument("--gold-root", type=Path, help="Root of generated task directories containing gold_skills.json")
    m_build.add_argument("--output", type=Path, required=True, help="Output JSON path")

    m_cross = sub.add_parser("cross-scale", help="Cross-scale retrieval scoring")
    m_cross.add_argument("--scales", type=int, nargs="+", required=True)
    m_cross.add_argument("--graph-dir", type=Path, required=True)
    m_cross.add_argument("--query-manifest", type=Path, required=True)
    m_cross.add_argument("--output", type=Path, required=True)

    m_edit = sub.add_parser("edit-effect", help="Cold-vs-edited graph comparison")
    m_edit.add_argument("--cold-graph", type=Path, required=True)
    m_edit.add_argument("--edited-graph", type=Path, required=True)
    m_edit.add_argument("--query-manifest", type=Path, required=True)
    m_edit.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.mode == "build-manifest":
        build_manifest(args.trials_dir, args.output, gold_root=getattr(args, 'gold_root', None))
    elif args.mode == "cross-scale":
        cross_scale(args.scales, args.graph_dir, args.query_manifest, args.output)
    elif args.mode == "edit-effect":
        edit_effect(args.cold_graph, args.edited_graph, args.query_manifest, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())