#!/usr/bin/env python3
"""Build ALFWorld-aligned, unlabeled multi-source SkillsBench candidates.

This stage is deliberately label-free. It combines retrieval sources for
later 2/1/0 judging, but never converts an unreviewed candidate into a negative.
Official SkillsBench tasks are outside the source allowlist and are not read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GOS_SRC = ROOT / "src/GraphOfSkills_NCF"
SCRIPTS = ROOT / "scripts/skillbench_ncf"
for path in (GOS_SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gos.ncf.models import NCFConfig, NeuMF  # noqa: E402
from source_policy import assert_training_source, load_policy  # noqa: E402


QUERIES = ROOT / "data/skillbench_ncf/expanded_queries_v2/queries.jsonl"
QUERY_EMBEDDINGS = ROOT / "data/skillbench_ncf/expanded_pairs_v2_balanced/query_embeddings.json"
EVIDENCE = ROOT / "data/skillbench_ncf/evidence_v1/skill_evidence_cards.jsonl"
GRAPH = ROOT / ".runtime/skillbench_ncf/sources/skillgraph_1000.json"
BASE_SKILLS = ROOT / ".runtime/skillbench_ncf/sources/skillgraph_1000.embeddings.json"
HYBRID_SKILLS = ROOT / "data/skillbench_ncf/enriched_skill_embeddings_v3/hybrid_skill_embeddings.json"
FROZEN_NCF = ROOT / ".runtime/skillbench_ncf/models/neumf_expanded_v2_balanced/neumf_graded.pt"
OUTPUT_DIR = ROOT / "data/skillbench_ncf/candidate_pool_v4"
REPORT = ROOT / "artifacts/skillbench_ncf/manifests/candidate_pool_v4_report.json"
SAMPLE = ROOT / "artifacts/skillbench_ncf/manifests/candidate_pool_v4_review_sample.json"

BASE_TOP_K = 12
HYBRID_TOP_K = 12
NCF_TOP_K = 8
GRAPH_NEW_CAP = 8
EASY_PROBES = 4
GRAPH_EDGE_TYPES = {"depends_on", "composes_with"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def embedding_items(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("items", value)


def distribution(values: list[int]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "min": int(data.min()),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "max": int(data.max()),
    }


@torch.no_grad()
def score_all_ncf(query_matrix: np.ndarray, skill_matrix: np.ndarray) -> np.ndarray:
    """Score all pairs while reusing the model's task/skill projections."""
    model = NeuMF(
        NCFConfig(input_dim=3072, gmf_latent_dim=16, mlp_layer_sizes=(64, 32, 16))
    )
    model.load_state_dict(torch.load(FROZEN_NCF, map_location="cpu", weights_only=True))
    model.eval()
    queries = torch.from_numpy(query_matrix.astype(np.float32, copy=False))
    skills = torch.from_numpy(skill_matrix.astype(np.float32, copy=False))
    gmf_skills = model.gmf_skill_projection(skills)
    mlp_skills = model.mlp_skill_projection(skills)
    result = np.empty((len(query_matrix), len(skill_matrix)), dtype=np.float32)
    for start in range(0, len(queries), 32):
        task = queries[start : start + 32]
        gmf_task = model.gmf_task_projection(task)
        mlp_task = model.mlp_task_projection(task)
        count = len(task)
        gmf = gmf_task[:, None, :] * gmf_skills[None, :, :]
        left = mlp_task[:, None, :].expand(count, len(skills), -1)
        right = mlp_skills[None, :, :].expand(count, len(skills), -1)
        mlp = model.mlp_tower(torch.cat((left, right), dim=-1))
        result[start : start + count] = model.output(
            torch.cat((gmf, mlp), dim=-1)
        ).squeeze(-1).numpy()
    return result


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
    evidence = load_jsonl(EVIDENCE)
    cards = {str(row["skill_id"]): row for row in evidence}
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    query_cache = json.loads(QUERY_EMBEDDINGS.read_text(encoding="utf-8"))["items"]
    base = embedding_items(BASE_SKILLS)
    hybrid = embedding_items(HYBRID_SKILLS)
    skill_ids = sorted(cards)
    if len(queries) != 3650 or len(skill_ids) != 1000:
        raise AssertionError("locked V4 inputs changed unexpectedly")
    if (
        set(skill_ids) != set(base)
        or set(skill_ids) != set(hybrid)
        or set(skill_ids) != set(graph["nodes"])
    ):
        raise AssertionError("skill IDs differ across evidence, graph, and embeddings")
    if any(row.get("official_skillsbench_task_used") for row in queries):
        raise AssertionError("official SkillsBench task content reached candidates")

    query_matrix = np.asarray(
        [query_cache[row["query_id"]]["embedding"] for row in queries], dtype=np.float32
    )
    base_matrix = np.asarray(
        [base[skill_id]["embedding"] for skill_id in skill_ids], dtype=np.float32
    )
    hybrid_matrix = np.asarray(
        [hybrid[skill_id]["embedding"] for skill_id in skill_ids], dtype=np.float32
    )
    if query_matrix.shape[1:] != (3072,) or base_matrix.shape != hybrid_matrix.shape:
        raise AssertionError("unexpected embedding dimensions")
    base_scores = normalized(query_matrix) @ normalized(base_matrix).T
    hybrid_scores = normalized(query_matrix) @ normalized(hybrid_matrix).T
    ncf_scores = score_all_ncf(query_matrix, base_matrix)

    id_to_index = {skill_id: index for index, skill_id in enumerate(skill_ids)}
    skill_id_array = np.asarray(skill_ids)
    eligible = {
        split: np.asarray(
            [id_to_index[s] for s in skill_ids if cards[s]["split"] == split],
            dtype=np.int64,
        )
        for split in ("train", "dev")
    }
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

    output_rows: list[dict[str, Any]] = []
    candidates_by_query: dict[str, list[dict[str, Any]]] = {}
    source_pair_counts: Counter[str] = Counter()
    exposure: Counter[str] = Counter()
    exposure_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_sizes: list[int] = []
    retrieval_hits = Counter()
    rank_values: dict[str, list[int]] = defaultdict(list)
    source_anchor_added = 0
    graph_new_counts: list[int] = []

    for query_index, query in enumerate(queries):
        split = str(query["internal_family_split"])
        positive = str(query["positive_skill_id"])
        positive_family = cards[positive]["family_id"]
        pool = eligible[split]
        merged: dict[str, dict[str, Any]] = {}

        def add_ranked(source: str, scores: np.ndarray, top_k: int) -> None:
            order = pool[np.lexsort((skill_id_array[pool], -scores[pool]))]
            positive_position = int(
                np.flatnonzero(order == id_to_index[positive])[0]
            ) + 1
            rank_values[source].append(positive_position)
            retrieval_hits[f"{source}@{top_k}"] += int(positive_position <= top_k)
            for rank, index in enumerate(order[:top_k], start=1):
                skill_id = skill_ids[int(index)]
                row = merged.setdefault(
                    skill_id,
                    {"skill_id": skill_id, "candidate_sources": [], "graph_paths": []},
                )
                row["candidate_sources"].append(source)
                row[f"{source}_rank"] = rank
                row[f"{source}_score"] = float(scores[index])

        add_ranked("base_cosine", base_scores[query_index], BASE_TOP_K)
        add_ranked("hybrid_cosine", hybrid_scores[query_index], HYBRID_TOP_K)
        add_ranked("v2_ncf_hard", ncf_scores[query_index], NCF_TOP_K)
        retrieved_before_graph = positive in merged

        seed_priority = sorted(
            merged,
            key=lambda sid: (
                min(
                    merged[sid].get("base_cosine_rank", 10_000),
                    merged[sid].get("hybrid_cosine_rank", 10_000),
                    merged[sid].get("v2_ncf_hard_rank", 10_000),
                ),
                sid,
            ),
        )
        graph_new = 0
        for seed_id in seed_priority:
            for neighbor in adjacency.get(seed_id, []):
                skill_id = neighbor["skill_id"]
                if cards[skill_id]["split"] != split:
                    continue
                if skill_id not in merged and graph_new >= GRAPH_NEW_CAP:
                    continue
                if skill_id not in merged:
                    merged[skill_id] = {
                        "skill_id": skill_id,
                        "candidate_sources": [],
                        "graph_paths": [],
                    }
                    graph_new += 1
                row = merged[skill_id]
                if "graph_one_hop" not in row["candidate_sources"]:
                    row["candidate_sources"].append("graph_one_hop")
                path = {key: value for key, value in neighbor.items() if key != "skill_id"}
                path["reached_from"] = seed_id
                if path not in row["graph_paths"]:
                    row["graph_paths"].append(path)
        graph_new_counts.append(graph_new)
        retrieved_before_anchor = positive in merged
        retrieval_hits["retrieved_union_before_graph"] += int(retrieved_before_graph)
        retrieval_hits["retrieved_union_after_graph"] += int(retrieved_before_anchor)

        easy_eligible = [
            skill_id
            for skill_id in skill_ids
            if cards[skill_id]["split"] == split
            and skill_id not in merged
            and cards[skill_id]["family_id"] != positive_family
            and skill_id not in any_neighbors.get(positive, set())
            and skill_id != positive
        ]
        easy_eligible.sort(
            key=lambda sid: (
                hashlib.sha256(f"{query['query_id']}::{sid}".encode()).hexdigest(),
                sid,
            )
        )
        for rank, skill_id in enumerate(easy_eligible[:EASY_PROBES], start=1):
            merged[skill_id] = {
                "skill_id": skill_id,
                "candidate_sources": ["deterministic_easy_probe"],
                "deterministic_easy_probe_rank": rank,
                "graph_paths": [],
            }

        if positive not in merged:
            merged[positive] = {
                "skill_id": positive,
                "candidate_sources": ["source_anchor"],
                "graph_paths": [],
            }
            source_anchor_added += 1

        rows_for_query = []
        for skill_id, item in sorted(merged.items()):
            sources = sorted(set(item["candidate_sources"]))
            row = {
                "schema_version": "skillbench_ncf.candidate_pair.v4",
                "pair_id": f"{query['query_id']}::{skill_id}",
                "query_id": query["query_id"],
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
            rows_for_query.append(row)
            output_rows.append(row)
            exposure[skill_id] += 1
            source_pair_counts.update(sources)
            for source in sources:
                exposure_by_source[source][skill_id] += 1
        candidates_by_query[query["query_id"]] = rows_for_query
        candidate_sizes.append(len(rows_for_query))

    if any(row["label"] is not None or not row["candidate_only"] for row in output_rows):
        raise AssertionError("candidate construction assigned a training label")
    if len({row["query_id"] for row in output_rows}) != len(queries):
        raise AssertionError("candidate construction lost queries")
    if any(
        not any(row["known_source_pair"] for row in rows)
        for rows in candidates_by_query.values()
    ):
        raise AssertionError("a query lost its provenance source pair")

    output_path = OUTPUT_DIR / "candidates.jsonl"
    dump_jsonl(output_path, output_rows)
    sample_queries = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        by_type[query["source_type"]].append(query)
    for source_type in sorted(by_type):
        selected = sorted(
            by_type[source_type],
            key=lambda row: (
                hashlib.sha256(row["query_id"].encode()).hexdigest(),
                row["query_id"],
            ),
        )[:4]
        for query in selected:
            candidates = candidates_by_query[query["query_id"]]
            sample_queries.append(
                {
                    "query_id": query["query_id"],
                    "query_text": query["query_text"],
                    "source_type": source_type,
                    "source_skill_id": query["positive_skill_id"],
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                }
            )
    dump_json(
        SAMPLE,
        {
            "schema_version": "skillbench_ncf.candidate_pool_review.v4",
            "warning": "candidate_only=true and label=null; inclusion does not imply irrelevance",
            "queries": sample_queries,
        },
    )

    query_count = len(queries)
    report = {
        "schema_version": "skillbench_ncf.candidate_pool_report.v4",
        "status": "complete",
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "alignment_with_alfworld_v3": {
            "shared_sources": [
                "context_cosine",
                "full_or_alternate_context_cosine",
                "frozen_ncf_hard",
                "graph_one_hop",
            ],
            "skillsbench_mapping": [
                "base_cosine",
                "hybrid_cosine",
                "v2_ncf_hard",
                "graph_one_hop",
            ],
            "labels_assigned_at_candidate_stage": False,
        },
        "configuration": {
            "base_cosine_top_k": BASE_TOP_K,
            "hybrid_cosine_top_k": HYBRID_TOP_K,
            "frozen_v2_ncf_top_k": NCF_TOP_K,
            "graph_new_candidate_cap": GRAPH_NEW_CAP,
            "graph_edge_types": sorted(GRAPH_EDGE_TYPES),
            "deterministic_easy_probes": EASY_PROBES,
            "same_family_split_required": True,
        },
        "counts": {
            "queries": query_count,
            "candidate_pairs": len(output_rows),
            "skills_exposed": len(exposure),
            "source_anchor_added": source_anchor_added,
            "candidate_pairs_by_source": dict(sorted(source_pair_counts.items())),
        },
        "candidate_pool_size": distribution(candidate_sizes),
        "graph_new_candidates_per_query": distribution(graph_new_counts),
        "source_retrieval": {
            key: {"hits": int(value), "recall": value / query_count}
            for key, value in sorted(retrieval_hits.items())
        },
        "positive_rank": {
            source: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "recall@20": float(np.mean(np.asarray(values) <= 20)),
            }
            for source, values in sorted(rank_values.items())
        },
        "skill_exposure": {
            "distribution": distribution(list(exposure.values())),
            "zero_exposure_skills": sorted(set(skill_ids) - set(exposure)),
            "top_20": [
                {"skill_id": skill_id, "queries": count}
                for skill_id, count in exposure.most_common(20)
            ],
            "top_5_by_candidate_source": {
                source: [
                    {
                        "skill_id": skill_id,
                        "queries": count,
                        "query_share": count / query_count,
                    }
                    for skill_id, count in counts.most_common(5)
                ]
                for source, counts in sorted(exposure_by_source.items())
            },
        },
        "label_contract": {
            "all_rows_candidate_only": True,
            "all_labels_null": True,
            "known_source_pair_is_provenance_not_a_judged_candidate_label": True,
            "next_stage": "template-level 2/1/0/uncertain evidence judge",
            "judge_ready": False,
            "blocking_audit": (
                "frozen-NCF-only candidates show extreme global exposure; "
                "template deduplication or frequency control is required before paid judging"
            ),
        },
        "inputs_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in paths
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
