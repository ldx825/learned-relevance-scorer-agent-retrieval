#!/usr/bin/env python3
"""Build a Judge-preparation pool with exposure-balanced frozen-NCF errors."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/skillbench_ncf"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_candidate_pool_v4 import (  # noqa: E402
    BASE_SKILLS,
    BASE_TOP_K,
    EASY_PROBES,
    EVIDENCE,
    FROZEN_NCF,
    GRAPH,
    GRAPH_EDGE_TYPES,
    GRAPH_NEW_CAP,
    HYBRID_SKILLS,
    HYBRID_TOP_K,
    QUERIES,
    QUERY_EMBEDDINGS,
    distribution,
    dump_json,
    dump_jsonl,
    embedding_items,
    load_jsonl,
    normalized,
    score_all_ncf,
    sha256_file,
)
from source_policy import assert_training_source, load_policy  # noqa: E402


RAW_REPORT = ROOT / "artifacts/skillbench_ncf/manifests/candidate_pool_v4_report.json"
OUTPUT_DIR = ROOT / "data/skillbench_ncf/candidate_pool_v4_balanced"
REPORT = ROOT / "artifacts/skillbench_ncf/manifests/candidate_pool_v4_balanced_report.json"
SAMPLE = ROOT / "artifacts/skillbench_ncf/manifests/candidate_pool_v4_balanced_review_sample.json"
NCF_SEARCH_K = 128
NCF_PER_QUERY = 4
NCF_EXPOSURE_FRACTION = 0.05


def main() -> int:
    policy = load_policy()
    paths = [
        QUERIES,
        QUERY_EMBEDDINGS,
        EVIDENCE,
        GRAPH,
        BASE_SKILLS,
        HYBRID_SKILLS,
        FROZEN_NCF,
    ]
    for path in paths:
        assert_training_source(path, policy)

    queries = sorted(load_jsonl(QUERIES), key=lambda row: row["query_id"])
    query_by_id = {row["query_id"]: row for row in queries}
    cards = {row["skill_id"]: row for row in load_jsonl(EVIDENCE)}
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    query_cache = json.loads(QUERY_EMBEDDINGS.read_text(encoding="utf-8"))["items"]
    base = embedding_items(BASE_SKILLS)
    hybrid = embedding_items(HYBRID_SKILLS)
    skill_ids = sorted(cards)
    skill_id_array = np.asarray(skill_ids)
    id_to_index = {skill_id: index for index, skill_id in enumerate(skill_ids)}
    query_matrix = np.asarray(
        [query_cache[row["query_id"]]["embedding"] for row in queries], dtype=np.float32
    )
    base_matrix = np.asarray(
        [base[skill_id]["embedding"] for skill_id in skill_ids], dtype=np.float32
    )
    hybrid_matrix = np.asarray(
        [hybrid[skill_id]["embedding"] for skill_id in skill_ids], dtype=np.float32
    )
    base_scores = normalized(query_matrix) @ normalized(base_matrix).T
    hybrid_scores = normalized(query_matrix) @ normalized(hybrid_matrix).T
    ncf_scores = score_all_ncf(query_matrix, base_matrix)
    eligible = {
        split: np.asarray(
            [id_to_index[s] for s in skill_ids if cards[s]["split"] == split],
            dtype=np.int64,
        )
        for split in ("train", "dev")
    }
    split_query_counts = Counter(row["internal_family_split"] for row in queries)
    exposure_caps = {
        split: math.ceil(count * NCF_EXPOSURE_FRACTION)
        for split, count in split_query_counts.items()
    }

    base_orders: dict[str, np.ndarray] = {}
    hybrid_orders: dict[str, np.ndarray] = {}
    ncf_orders: dict[str, np.ndarray] = {}
    for index, query in enumerate(queries):
        pool = eligible[query["internal_family_split"]]
        base_orders[query["query_id"]] = pool[
            np.lexsort((skill_id_array[pool], -base_scores[index, pool]))
        ]
        hybrid_orders[query["query_id"]] = pool[
            np.lexsort((skill_id_array[pool], -hybrid_scores[index, pool]))
        ]
        ncf_orders[query["query_id"]] = pool[
            np.lexsort((skill_id_array[pool], -ncf_scores[index, pool]))
        ]

    # Allocate hard candidates in a hash-stable order so alphabetical query
    # order cannot decide which skills consume the global exposure budget.
    selected_ncf: dict[str, list[tuple[int, int]]] = {}
    ncf_exposure: dict[str, Counter[str]] = defaultdict(Counter)
    selection_shortfalls = 0
    selection_ranks: list[int] = []
    allocation_order = sorted(
        queries,
        key=lambda row: (
            hashlib.sha256(row["query_id"].encode()).hexdigest(),
            row["query_id"],
        ),
    )
    for query in allocation_order:
        query_id = query["query_id"]
        split = query["internal_family_split"]
        core = {
            int(index)
            for index in np.concatenate(
                (base_orders[query_id][:BASE_TOP_K], hybrid_orders[query_id][:HYBRID_TOP_K])
            )
        }
        chosen: list[tuple[int, int]] = []
        for raw_rank, index in enumerate(ncf_orders[query_id][:NCF_SEARCH_K], start=1):
            index = int(index)
            skill_id = skill_ids[index]
            if index in core or ncf_exposure[split][skill_id] >= exposure_caps[split]:
                continue
            chosen.append((index, raw_rank))
            ncf_exposure[split][skill_id] += 1
            selection_ranks.append(raw_rank)
            if len(chosen) == NCF_PER_QUERY:
                break
        selected_ncf[query_id] = chosen
        selection_shortfalls += int(len(chosen) < NCF_PER_QUERY)

    adjacency: dict[str, list[dict[str, str]]] = defaultdict(list)
    any_neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in graph["edges"]:
        source, target = str(edge["source"]), str(edge["target"])
        any_neighbors[source].add(target)
        any_neighbors[target].add(source)
        if edge["type"] not in GRAPH_EDGE_TYPES:
            continue
        record = {
            "edge_type": str(edge["type"]),
            "edge_source": source,
            "edge_target": target,
        }
        adjacency[source].append({**record, "skill_id": target})
        adjacency[target].append({**record, "skill_id": source})
    for rows in adjacency.values():
        rows.sort(key=lambda row: (row["edge_type"], row["skill_id"]))

    rows: list[dict[str, Any]] = []
    by_query: dict[str, list[dict[str, Any]]] = {}
    candidate_sizes: list[int] = []
    exposure: Counter[str] = Counter()
    exposure_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Counter[str] = Counter()
    anchors = 0
    graph_new_counts: list[int] = []
    union_source_hits = 0

    for query_index, query in enumerate(queries):
        query_id = query["query_id"]
        split = query["internal_family_split"]
        positive = query["positive_skill_id"]
        positive_family = cards[positive]["family_id"]
        merged: dict[str, dict[str, Any]] = {}

        def add(source: str, index: int, rank: int, score: float) -> None:
            skill_id = skill_ids[index]
            item = merged.setdefault(
                skill_id,
                {"candidate_sources": [], "graph_paths": []},
            )
            item["candidate_sources"].append(source)
            item[f"{source}_rank"] = rank
            item[f"{source}_score"] = score

        for rank, index in enumerate(base_orders[query_id][:BASE_TOP_K], start=1):
            add("base_cosine", int(index), rank, float(base_scores[query_index, index]))
        for rank, index in enumerate(hybrid_orders[query_id][:HYBRID_TOP_K], start=1):
            add("hybrid_cosine", int(index), rank, float(hybrid_scores[query_index, index]))
        for selection_rank, (index, raw_rank) in enumerate(selected_ncf[query_id], start=1):
            add("v2_ncf_balanced_hard", index, selection_rank, float(ncf_scores[query_index, index]))
            merged[skill_ids[index]]["v2_ncf_raw_rank"] = raw_rank

        seed_ids = sorted(
            merged,
            key=lambda sid: (
                min(
                    merged[sid].get("base_cosine_rank", 10_000),
                    merged[sid].get("hybrid_cosine_rank", 10_000),
                    merged[sid].get("v2_ncf_balanced_hard_rank", 10_000),
                ),
                sid,
            ),
        )
        graph_new = 0
        for seed_id in seed_ids:
            for neighbor in adjacency.get(seed_id, []):
                skill_id = neighbor["skill_id"]
                if cards[skill_id]["split"] != split:
                    continue
                if skill_id not in merged and graph_new >= GRAPH_NEW_CAP:
                    continue
                if skill_id not in merged:
                    merged[skill_id] = {"candidate_sources": [], "graph_paths": []}
                    graph_new += 1
                item = merged[skill_id]
                if "graph_one_hop" not in item["candidate_sources"]:
                    item["candidate_sources"].append("graph_one_hop")
                path = {key: value for key, value in neighbor.items() if key != "skill_id"}
                path["reached_from"] = seed_id
                if path not in item["graph_paths"]:
                    item["graph_paths"].append(path)
        graph_new_counts.append(graph_new)
        union_source_hits += int(positive in merged)

        easy = [
            skill_id
            for skill_id in skill_ids
            if cards[skill_id]["split"] == split
            and skill_id not in merged
            and skill_id != positive
            and cards[skill_id]["family_id"] != positive_family
            and skill_id not in any_neighbors.get(positive, set())
        ]
        easy.sort(
            key=lambda sid: (
                hashlib.sha256(f"{query_id}::{sid}".encode()).hexdigest(),
                sid,
            )
        )
        for rank, skill_id in enumerate(easy[:EASY_PROBES], start=1):
            merged[skill_id] = {
                "candidate_sources": ["deterministic_easy_probe"],
                "deterministic_easy_probe_rank": rank,
                "graph_paths": [],
            }

        if positive not in merged:
            merged[positive] = {
                "candidate_sources": ["source_anchor"],
                "graph_paths": [],
            }
            anchors += 1

        query_rows = []
        for skill_id, item in sorted(merged.items()):
            sources = sorted(set(item["candidate_sources"]))
            row = {
                "schema_version": "skillbench_ncf.balanced_candidate_pair.v4",
                "pair_id": f"{query_id}::{skill_id}",
                "query_id": query_id,
                "query_text": query["query_text"],
                "query_source_type": query["source_type"],
                "internal_family_split": split,
                "source_skill_id": positive,
                "candidate_skill_id": skill_id,
                "candidate_family_id": cards[skill_id]["family_id"],
                "candidate_description": cards[skill_id]["description"],
                "candidate_sources": sources,
                "graph_paths": item["graph_paths"],
                "known_source_pair": skill_id == positive,
                "candidate_only": True,
                "label": None,
                "official_skillsbench_task_used": False,
                **{
                    key: value
                    for key, value in item.items()
                    if key.endswith("_rank") or key.endswith("_score")
                },
            }
            rows.append(row)
            query_rows.append(row)
            exposure[skill_id] += 1
            source_counts.update(sources)
            for source in sources:
                exposure_by_source[source][skill_id] += 1
        by_query[query_id] = query_rows
        candidate_sizes.append(len(query_rows))

    if selection_shortfalls:
        raise AssertionError(f"{selection_shortfalls} queries lack four balanced NCF candidates")
    if any(row["label"] is not None for row in rows):
        raise AssertionError("balanced candidate construction assigned labels")
    if len(exposure) != 1000:
        raise AssertionError("balanced candidate pool lost skill coverage")
    for split, counts in ncf_exposure.items():
        if counts and max(counts.values()) > exposure_caps[split]:
            raise AssertionError("NCF exposure cap was exceeded")

    output_path = OUTPUT_DIR / "candidates.jsonl"
    dump_jsonl(output_path, rows)
    sample_ids = [
        row["query_id"]
        for row in sorted(
            queries,
            key=lambda row: (
                hashlib.sha256(row["query_id"].encode()).hexdigest(),
                row["query_id"],
            ),
        )[:20]
    ]
    dump_json(
        SAMPLE,
        {
            "schema_version": "skillbench_ncf.balanced_candidate_review.v4",
            "warning": "candidate_only=true and label=null; this is not judged training data",
            "queries": [
                {
                    "query_id": query_id,
                    "query_text": query_by_id[query_id]["query_text"],
                    "source_skill_id": query_by_id[query_id]["positive_skill_id"],
                    "candidates": by_query[query_id],
                }
                for query_id in sample_ids
            ],
        },
    )

    raw_report = json.loads(RAW_REPORT.read_text(encoding="utf-8"))
    max_balanced_share = max(
        count / split_query_counts[split]
        for split, counts in ncf_exposure.items()
        for count in counts.values()
    )
    report = {
        "schema_version": "skillbench_ncf.balanced_candidate_pool_report.v4",
        "status": "complete",
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "configuration": {
            "base_cosine_top_k": BASE_TOP_K,
            "hybrid_cosine_top_k": HYBRID_TOP_K,
            "frozen_ncf_search_top_k": NCF_SEARCH_K,
            "balanced_ncf_candidates_per_query": NCF_PER_QUERY,
            "ncf_max_exposure_fraction_per_split": NCF_EXPOSURE_FRACTION,
            "ncf_exposure_cap_by_split": exposure_caps,
            "graph_new_candidate_cap": GRAPH_NEW_CAP,
            "graph_edge_types": sorted(GRAPH_EDGE_TYPES),
            "deterministic_easy_probes": EASY_PROBES,
        },
        "counts": {
            "queries": len(queries),
            "candidate_pairs": len(rows),
            "skills_exposed": len(exposure),
            "source_anchor_added": anchors,
            "candidate_pairs_by_source": dict(sorted(source_counts.items())),
            "ncf_selection_shortfall_queries": selection_shortfalls,
        },
        "candidate_pool_size": distribution(candidate_sizes),
        "graph_new_candidates_per_query": distribution(graph_new_counts),
        "source_retrieval_before_anchor": {
            "hits": union_source_hits,
            "recall": union_source_hits / len(queries),
        },
        "ncf_exposure_audit": {
            "raw_v2_top_skill_query_share": raw_report["skill_exposure"]["top_5_by_candidate_source"]["v2_ncf_hard"][0]["query_share"],
            "balanced_max_query_share_within_split": max_balanced_share,
            "selected_raw_rank": distribution(selection_ranks),
            "top_20": {
                split: [
                    {
                        "skill_id": skill_id,
                        "queries": count,
                        "split_query_share": count / split_query_counts[split],
                    }
                    for skill_id, count in counts.most_common(20)
                ]
                for split, counts in sorted(ncf_exposure.items())
            },
        },
        "skill_exposure": {
            "distribution": distribution(list(exposure.values())),
            "zero_exposure_skills": sorted(set(skill_ids) - set(exposure)),
            "top_5_by_candidate_source": {
                source: [
                    {"skill_id": skill_id, "queries": count}
                    for skill_id, count in counts.most_common(5)
                ]
                for source, counts in sorted(exposure_by_source.items())
            },
        },
        "label_contract": {
            "all_rows_candidate_only": True,
            "all_labels_null": True,
            "same_family_split_required": True,
            "judge_ready_for_small_pilot": True,
            "not_authorized_for_api_submission": True,
            "next_stage": "construct a zero-API template/batch manifest before requesting Judge authorization",
        },
        "inputs_sha256": {
            **{str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
            str(RAW_REPORT.relative_to(ROOT)): sha256_file(RAW_REPORT),
        },
        "private_output": str(output_path.relative_to(ROOT)),
        "private_output_sha256": sha256_file(output_path),
        "review_sample": str(SAMPLE.relative_to(ROOT)),
    }
    dump_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
