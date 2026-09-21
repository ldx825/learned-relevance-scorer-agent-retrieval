#!/usr/bin/env python3
"""Compare frozen SkillsBench hybrid plans with cosine on curated eval skills.

This script is evaluation-only. It reads task-visible plan queries and the
curated skills shipped with each official task, but it does not train or tune
the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def gold_skills(task_dir: Path) -> set[str]:
    root = task_dir / "environment" / "skills"
    if not root.is_dir():
        return set()
    return {
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    }


def reciprocal_rank(ranking: list[str], gold: set[str]) -> float:
    return next(
        (1.0 / rank for rank, skill_id in enumerate(ranking, 1) if skill_id in gold),
        0.0,
    )


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    count = len(rows)
    return {
        "queries": count,
        "ret_at_1": sum(row["hit_at_1"] for row in rows) / count,
        "ret_at_3": sum(row["hit_at_3"] for row in rows) / count,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / count,
        "mean_gold_hits_at_3": sum(row["gold_hits_at_3"] for row in rows) / count,
        "precision_at_3": sum(row["gold_hits_at_3"] for row in rows) / (3 * count),
        "gold_recall_at_3": sum(row["gold_recall_at_3"] for row in rows) / count,
    }


def ranking_overlap(left: list[str], right: list[str]) -> float:
    """Return set Jaccard overlap for two fixed-width rankings."""
    union = set(left).union(right)
    return len(set(left).intersection(right)) / len(union) if union else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--skill-embeddings", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plans = load_json(args.plans)
    cache_payload = load_json(args.query_cache)
    skill_payload = load_json(args.skill_embeddings)
    # The original SkillDAG cache is a flat skill-id map.  The enriched V5
    # cache stores the same entries under ``items`` alongside provenance.
    skill_items = skill_payload.get("items", skill_payload)
    if not isinstance(skill_items, dict):
        raise ValueError("skill embeddings must be a skill-id map or contain items")
    vectors = cache_payload["vectors"]
    skill_ids = sorted(skill_items)
    skill_matrix = np.asarray(
        [skill_items[skill_id]["embedding"] for skill_id in skill_ids],
        dtype=np.float32,
    )
    skill_matrix /= np.linalg.norm(skill_matrix, axis=1, keepdims=True).clip(1e-12)
    catalog = set(skill_ids)

    rows: list[dict] = []
    missing_gold: dict[str, list[str]] = {}
    task_unions: list[dict] = []
    for task_id, task_plan in plans["tasks"].items():
        gold = gold_skills(args.tasks_root / task_id)
        missing = sorted(gold - catalog)
        if missing:
            missing_gold[task_id] = missing
        cosine_union: set[str] = set()
        hybrid_union: set[str] = set()
        for view in task_plan["views"]:
            query = view["query"]
            key = hashlib.sha256(query.encode("utf-8")).hexdigest()
            query_vector = np.asarray(vectors[key], dtype=np.float32)
            query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
            scores = skill_matrix @ query_vector
            order = sorted(
                range(len(skill_ids)),
                key=lambda index: (-float(scores[index]), skill_ids[index]),
            )[:3]
            cosine_ranking = [skill_ids[index] for index in order]
            hybrid_ranking = [match["skill_id"] for match in view["matches"]]
            cosine_union.update(cosine_ranking)
            hybrid_union.update(hybrid_ranking)
            for method, ranking in (
                ("cosine", cosine_ranking),
                ("hybrid", hybrid_ranking),
            ):
                hits = gold.intersection(ranking)
                rows.append(
                    {
                        "task_id": task_id,
                        "view": view["name"],
                        "method": method,
                        "ranking": ranking,
                        "gold_skills": sorted(gold),
                        "hit_at_1": bool(ranking and ranking[0] in gold),
                        "hit_at_3": bool(hits),
                        "reciprocal_rank": reciprocal_rank(ranking, gold),
                        "gold_hits_at_3": len(hits),
                        "gold_recall_at_3": len(hits) / len(gold) if gold else 0.0,
                    }
                )
        task_unions.append(
            {
                "task_id": task_id,
                "gold_skills": sorted(gold),
                "cosine_hit": bool(gold.intersection(cosine_union)),
                "hybrid_hit": bool(gold.intersection(hybrid_union)),
                "cosine_gold_recall": (
                    len(gold.intersection(cosine_union)) / len(gold) if gold else 0.0
                ),
                "hybrid_gold_recall": (
                    len(gold.intersection(hybrid_union)) / len(gold) if gold else 0.0
                ),
            }
        )

    metrics = {}
    for scope, predicate in (
        ("full_task", lambda row: row["view"] == "full_task"),
        ("all_views", lambda row: True),
    ):
        metrics[scope] = {
            method: aggregate(
                [
                    row
                    for row in rows
                    if row["method"] == method and predicate(row)
                ]
            )
            for method in ("cosine", "hybrid")
        }

    cosine_rows = {
        (row["task_id"], row["view"]): row
        for row in rows
        if row["method"] == "cosine"
    }
    hybrid_rows = {
        (row["task_id"], row["view"]): row
        for row in rows
        if row["method"] == "hybrid"
    }
    flips = {"rescued": [], "harmed": []}
    for key, cosine_row in cosine_rows.items():
        hybrid_row = hybrid_rows[key]
        if not cosine_row["hit_at_3"] and hybrid_row["hit_at_3"]:
            flips["rescued"].append(
                {
                    "task_id": key[0],
                    "view": key[1],
                    "gold_skills": cosine_row["gold_skills"],
                    "cosine": cosine_row["ranking"],
                    "hybrid": hybrid_row["ranking"],
                }
            )
        elif cosine_row["hit_at_3"] and not hybrid_row["hit_at_3"]:
            flips["harmed"].append(
                {
                    "task_id": key[0],
                    "view": key[1],
                    "gold_skills": cosine_row["gold_skills"],
                    "cosine": cosine_row["ranking"],
                    "hybrid": hybrid_row["ranking"],
                }
            )

    paired_views = []
    for key, cosine_row in cosine_rows.items():
        hybrid_row = hybrid_rows[key]
        cosine_ranking = cosine_row["ranking"]
        hybrid_ranking = hybrid_row["ranking"]
        paired_views.append(
            {
                "task_id": key[0],
                "view": key[1],
                "exact_order_equal": cosine_ranking == hybrid_ranking,
                "top_k_set_equal": set(cosine_ranking) == set(hybrid_ranking),
                "jaccard": ranking_overlap(cosine_ranking, hybrid_ranking),
                "recall_delta": (
                    hybrid_row["gold_recall_at_3"] - cosine_row["gold_recall_at_3"]
                ),
                "rr_delta": (
                    hybrid_row["reciprocal_rank"] - cosine_row["reciprocal_rank"]
                ),
            }
        )

    differing_tasks = sorted(
        {row["task_id"] for row in paired_views if not row["exact_order_equal"]}
    )
    count = len(paired_views)
    disagreement = {
        "views": count,
        "exact_order_changed": sum(not row["exact_order_equal"] for row in paired_views),
        "top_k_membership_changed": sum(not row["top_k_set_equal"] for row in paired_views),
        "tasks_with_any_order_change": len(differing_tasks),
        "task_ids_with_any_order_change": differing_tasks,
        "mean_top_k_jaccard": (
            sum(row["jaccard"] for row in paired_views) / count if count else 1.0
        ),
        "gold_recall_better": sum(row["recall_delta"] > 0 for row in paired_views),
        "gold_recall_worse": sum(row["recall_delta"] < 0 for row in paired_views),
        "gold_recall_equal": sum(row["recall_delta"] == 0 for row in paired_views),
        "reciprocal_rank_better": sum(row["rr_delta"] > 0 for row in paired_views),
        "reciprocal_rank_worse": sum(row["rr_delta"] < 0 for row in paired_views),
        "reciprocal_rank_equal": sum(row["rr_delta"] == 0 for row in paired_views),
    }

    task_count = len(task_unions)
    report = {
        "schema_version": "skillbench_ncf.frozen_plan_eval.v1",
        "evaluation_only": True,
        "task_count": task_count,
        "gold_catalog_coverage": {
            "missing_by_task": missing_gold,
            "status": "pass" if not missing_gold else "fail",
        },
        "metrics": metrics,
        "task_union": {
            "cosine_hit_rate": sum(row["cosine_hit"] for row in task_unions)
            / task_count,
            "hybrid_hit_rate": sum(row["hybrid_hit"] for row in task_unions)
            / task_count,
            "cosine_gold_recall": sum(
                row["cosine_gold_recall"] for row in task_unions
            )
            / task_count,
            "hybrid_gold_recall": sum(
                row["hybrid_gold_recall"] for row in task_unions
            )
            / task_count,
        },
        "flips_at_3": flips,
        "ranking_disagreement": disagreement,
        "paired_views": paired_views,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("task_count", "gold_catalog_coverage", "metrics", "task_union", "ranking_disagreement", "flips_at_3")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
