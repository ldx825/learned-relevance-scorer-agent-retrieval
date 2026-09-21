#!/usr/bin/env python3
"""Replay the frozen ALFWorld Skill and Memory retrieval protocols.

This is the first ablation gate.  It performs no training and no API calls.
It fails closed unless the frozen cosine/deployed metrics exactly reproduce
their reference reports, then records pure full-pool NeuMF rankings for fair
model-architecture ablations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "src/GraphOfSkills_NCF",
    ROOT / "src/SkillDAG_NCF/src",
    ROOT / "scripts/memp",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gos.ncf.models import GMF, MLP, NCFConfig, NeuMF  # noqa: E402
from build_memory_query_only_candidates import signature, view_relation  # noqa: E402
from skilldag.hybrid_selection import select_diverse  # noqa: E402


DEFAULT_MANIFEST = ROOT / "configs/ablations/alfworld_ncf_v1.json"
DEFAULT_OUTPUT = ROOT / ".runtime/ablations/alfworld_ncf_v1/offline_replay"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve(entry: dict[str, str]) -> Path:
    return ROOT / entry["path"]


def normalized(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def normalize_scores(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return [0.0] * len(values) if std < 1e-8 else [(value - mean) / std for value in values]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))


def load_model(path: Path, model_type: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = {"gmf": GMF, "mlp": MLP, "neumf": NeuMF}
    if model_type not in classes:
        raise ValueError(f"unsupported model type: {model_type}")
    model = classes[model_type](NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def score_matrix(
    model: torch.nn.Module,
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    query_count, candidate_count = len(query_embeddings), len(candidate_embeddings)
    query_indices = np.repeat(np.arange(query_count), candidate_count)
    candidate_indices = np.tile(np.arange(candidate_count), query_count)
    chunks: list[np.ndarray] = []
    for start in range(0, len(query_indices), batch_size):
        selected = slice(start, start + batch_size)
        queries = torch.from_numpy(query_embeddings[query_indices[selected]]).float()
        candidates = torch.from_numpy(candidate_embeddings[candidate_indices[selected]]).float()
        chunks.append(model(queries, candidates).cpu().numpy())
    return np.concatenate(chunks).reshape(query_count, candidate_count)


def rank_descending(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")


def first_rank(ranking: list[str], gold: set[str], final_k: int) -> int | None:
    return next(
        (rank for rank, item_id in enumerate(ranking[:final_k], 1) if item_id in gold),
        None,
    )


def skill_metrics(rankings: list[list[str]], golds: list[set[str]], final_k: int) -> dict[str, Any]:
    ranks = [first_rank(ranking, gold, final_k) for ranking, gold in zip(rankings, golds)]
    count = len(ranks)
    return {
        "query_count": count,
        f"ret@{final_k}": sum(rank is not None for rank in ranks) / count,
        "ret@1": sum(rank == 1 for rank in ranks) / count,
        f"mrr@{final_k}": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / count,
    }


def assert_close(label: str, actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def replay_skill(manifest: dict[str, Any], output_dir: Path, batch_size: int) -> dict[str, Any]:
    spec = manifest["skill_alfworld"]
    reference = read_json(resolve(spec["reference_report"]))
    rows = reference["rows"]
    graph = read_json(resolve(spec["graph"]))
    skill_cache = read_json(resolve(spec["graph_embeddings"]))
    eval_cache = read_json(resolve(spec["evaluation_embeddings"]))
    entries = eval_cache["entries"]
    skill_ids = sorted(skill_cache)
    if len(skill_ids) != int(spec["offline_eval_contract"]["candidate_pool"]):
        raise AssertionError("Skill candidate pool changed")

    def text_embedding(text: str) -> np.ndarray:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entry = entries.get(key)
        if entry is None or entry.get("text") != text:
            raise KeyError(f"frozen evaluation embedding missing for: {text!r}")
        return np.asarray(entry["embedding"], dtype=np.float32)

    query_embeddings = np.stack([text_embedding(row["query"]) for row in rows])
    context_embeddings = np.stack([text_embedding(row["context"]) for row in rows])
    skill_embeddings = np.stack(
        [np.asarray(skill_cache[skill_id]["embedding"], dtype=np.float32) for skill_id in skill_ids]
    )
    learned_scores = {
        model_type: score_matrix(
            load_model(resolve(spec["frozen_checkpoints"][model_type]), model_type),
            context_embeddings,
            skill_embeddings,
            batch_size,
        )
        for model_type in ("gmf", "mlp", "neumf")
    }
    neumf_scores = learned_scores["neumf"]

    cosine_rankings: list[list[str]] = []
    pure_rankings: dict[str, list[list[str]]] = {
        "gmf": [],
        "mlp": [],
        "neumf": [],
    }
    top12_rerankings: dict[str, list[list[str]]] = {
        "gmf": [],
        "mlp": [],
        "neumf": [],
    }
    deployed_rankings: list[list[str]] = []
    prior_corrected_rankings: list[list[str]] = []
    calibrated_fusion_rankings: list[list[str]] = []
    output_rows: list[dict[str, Any]] = []
    calibration = read_json(resolve(spec["calibration"]))
    alpha = float(calibration.get("alpha", 0.25))
    priors = calibration.get("skill_logit_prior", {})
    graph_nodes = graph["nodes"]
    graph_edges = graph.get("edges", [])

    for query_index, row in enumerate(rows):
        cosine_by_id = {
            skill_id: cosine(query_embeddings[query_index], skill_embeddings[skill_index])
            for skill_index, skill_id in enumerate(skill_ids)
        }
        cosine_ranking = sorted(skill_ids, key=lambda item_id: (-cosine_by_id[item_id], item_id))
        query_pure_rankings = {
            model_type: sorted(
                skill_ids,
                key=lambda item_id: (
                    -float(scores[query_index, skill_ids.index(item_id)]),
                    item_id,
                ),
            )
            for model_type, scores in learned_scores.items()
        }
        candidate_ids = cosine_ranking[:12]
        query_top12_rerankings = {
            model_type: sorted(
                candidate_ids,
                key=lambda item_id: (
                    -float(scores[query_index, skill_ids.index(item_id)]),
                    item_id,
                ),
            )
            for model_type, scores in learned_scores.items()
        }
        candidate_logits = [
            float(neumf_scores[query_index, skill_ids.index(skill_id)]) for skill_id in candidate_ids
        ]
        specific = [
            score - float(priors.get(skill_id, 0.0))
            for skill_id, score in zip(candidate_ids, candidate_logits)
        ]
        cosine_z = normalize_scores([cosine_by_id[skill_id] for skill_id in candidate_ids])
        ncf_z = normalize_scores(specific)
        prior_corrected_ranking = [
            skill_id
            for _, skill_id in sorted(
                zip(specific, candidate_ids),
                key=lambda item: (-item[0], item[1]),
            )
        ]
        candidates = [
            {
                "skill_id": skill_id,
                "description": graph_nodes[skill_id].get("description", ""),
                "cosine_score": cosine_by_id[skill_id],
                "ncf_logit": logit,
                "base_score": alpha * cz + (1.0 - alpha) * nz,
            }
            for skill_id, logit, cz, nz in zip(candidate_ids, candidate_logits, cosine_z, ncf_z)
        ]
        candidates.sort(key=lambda item: (-item["base_score"], item["skill_id"]))
        calibrated_fusion_ranking = [item["skill_id"] for item in candidates]
        selected = select_diverse(
            candidates,
            edges=graph_edges,
            cosine_safe_ids=set(cosine_ranking[:3]),
            top_k=5,
        )
        deployed_ranking = [item["skill_id"] for item in selected]
        cosine_rankings.append(cosine_ranking)
        for model_type, ranking in query_pure_rankings.items():
            pure_rankings[model_type].append(ranking)
        for model_type, ranking in query_top12_rerankings.items():
            top12_rerankings[model_type].append(ranking)
        deployed_rankings.append(deployed_ranking)
        prior_corrected_rankings.append(prior_corrected_ranking)
        calibrated_fusion_rankings.append(calibrated_fusion_ranking)
        output_rows.append(
            {
                "query_index": query_index,
                "task_id": row["task_id"],
                "phase": row["phase"],
                "query": row["query"],
                "gold": row["gold"],
                "query_cosine_full_pool": cosine_ranking,
                "pure_gmf_full_pool": query_pure_rankings["gmf"],
                "pure_mlp_full_pool": query_pure_rankings["mlp"],
                "pure_neumf_full_pool": query_pure_rankings["neumf"],
                "cosine_top12_then_gmf": query_top12_rerankings["gmf"],
                "cosine_top12_then_mlp": query_top12_rerankings["mlp"],
                "cosine_top12_then_neumf": query_top12_rerankings["neumf"],
                "cosine_top12_then_prior_corrected_neumf": prior_corrected_ranking,
                "cosine_top12_then_calibrated_fusion": calibrated_fusion_ranking,
                "deployed_v3_top5": deployed_ranking,
            }
        )

    # Exact top-k replay catches candidate, tie-break, calibration, and graph drift.
    for index, row in enumerate(rows):
        if cosine_rankings[index][:5] != row["skilldag_top_k"]:
            raise AssertionError(f"Skill cosine ranking drift at query {index}")
        if deployed_rankings[index] != row["skilldag_ncf_top_k"]:
            raise AssertionError(f"Skill deployed V3 ranking drift at query {index}")

    golds = [set(row["gold"]) for row in rows]
    methods = {
        "query_cosine_full_pool": skill_metrics(cosine_rankings, golds, 5),
        "content_gmf_full_pool": skill_metrics(pure_rankings["gmf"], golds, 5),
        "content_mlp_full_pool": skill_metrics(pure_rankings["mlp"], golds, 5),
        "content_neumf_full_pool": skill_metrics(pure_rankings["neumf"], golds, 5),
        "cosine_top12_then_gmf": skill_metrics(top12_rerankings["gmf"], golds, 5),
        "cosine_top12_then_mlp": skill_metrics(top12_rerankings["mlp"], golds, 5),
        "cosine_top12_then_neumf": skill_metrics(top12_rerankings["neumf"], golds, 5),
        "cosine_top12_then_prior_corrected_neumf": skill_metrics(
            prior_corrected_rankings, golds, 5
        ),
        "cosine_top12_then_calibrated_fusion": skill_metrics(
            calibrated_fusion_rankings, golds, 5
        ),
        "deployed_v3_cosine_ncf_graph": skill_metrics(deployed_rankings, golds, 5),
    }
    contract = spec["offline_eval_contract"]
    assert_close("skill cosine Ret@1", methods["query_cosine_full_pool"]["ret@1"], contract["local_cosine_ret1"])
    assert_close("skill cosine MRR", methods["query_cosine_full_pool"]["mrr@5"], contract["local_cosine_mrr"])
    assert_close("skill deployed Ret@1", methods["deployed_v3_cosine_ncf_graph"]["ret@1"], contract["deployed_v3_ret1"])
    assert_close("skill deployed MRR", methods["deployed_v3_cosine_ncf_graph"]["mrr@5"], contract["deployed_v3_mrr"])
    report = {
        "schema_version": "agent_skill_evolution.skill_offline_replay.v1",
        "evaluation_split": "frozen ALFWorld valid_seen phase queries",
        "official_rewards_or_action_traces_used": False,
        "candidate_pool": len(skill_ids),
        "final_k": 5,
        "metric_contract": "first member of the fixed semantic gold skill family, truncated at K",
        "methods": methods,
    }
    write_json(output_dir / "skill_metrics.json", report)
    write_jsonl(output_dir / "skill_rankings.jsonl", output_rows)
    return {
        "report": report,
        "query_ids": [f'{row["task_id"]}::{row["phase"]}::{index}' for index, row in enumerate(rows)],
        "candidate_ids": skill_ids,
        "gold": [row["gold"] for row in rows],
    }


def dcg(grades: Iterable[int]) -> float:
    return sum((2**int(grade) - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def memory_metrics(grades: np.ndarray, rankings: np.ndarray, final_k: int) -> dict[str, Any]:
    eligible = np.any(grades == 2, axis=1)
    hits: dict[int, list[float]] = {1: [], 5: [], 10: []}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    precisions: list[float] = []
    for query_index in range(len(grades)):
        ranked_grades = grades[query_index, rankings[query_index, :final_k]]
        if eligible[query_index]:
            for k in hits:
                hits[k].append(float(np.any(ranked_grades[: min(k, final_k)] == 2)))
            first = np.flatnonzero(ranked_grades == 2)
            reciprocal_ranks.append(1.0 / (int(first[0]) + 1) if len(first) else 0.0)
        ideal = np.sort(grades[query_index])[::-1][:final_k]
        denominator = dcg(ideal)
        ndcgs.append(dcg(ranked_grades) / denominator if denominator else 0.0)
        precisions.append(float(np.mean(ranked_grades > 0)))
    return {
        "query_count": int(len(grades)),
        "grade_2_eligible_query_count": int(np.sum(eligible)),
        "ret@1": float(np.mean(hits[1])),
        "ret@5": float(np.mean(hits[5])),
        "ret@10": float(np.mean(hits[10])),
        "mrr@10": float(np.mean(reciprocal_ranks)),
        "ndcg@10": float(np.mean(ndcgs)),
        "compatible_precision@10": float(np.mean(precisions)),
    }


def cosine_then_neumf(cosine_scores: np.ndarray, neumf_scores: np.ndarray, candidate_c: int) -> np.ndarray:
    cosine_candidates = rank_descending(cosine_scores)[:, :candidate_c]
    output = np.empty_like(cosine_candidates)
    for query_index, candidates in enumerate(cosine_candidates):
        rerank = np.argsort(-neumf_scores[query_index, candidates], kind="stable")
        output[query_index] = candidates[rerank]
    return output


def replay_memory(manifest: dict[str, Any], output_dir: Path, batch_size: int) -> dict[str, Any]:
    spec = manifest["memory_alfworld"]
    views = read_jsonl(resolve(spec["query_views"]))
    memories = read_jsonl(resolve(spec["memory_bank"]))
    with np.load(resolve(spec["arrays"]), allow_pickle=False) as payload:
        task_ids = payload["task_ids"].copy()
        all_task_embeddings = payload["task_embeddings"].copy()
        memory_embeddings = payload["memory_embeddings"].copy()
    task_index_by_id = {int(task_id): index for index, task_id in enumerate(task_ids)}
    selected_views = [
        row for row in views
        if row["model_split"] == "model_dev" and int(row["query_view_index"]) in task_index_by_id
    ]
    selected_views.sort(key=lambda row: int(row["query_view_index"]))
    selected_indices = [task_index_by_id[int(row["query_view_index"])] for row in selected_views]
    task_embeddings = all_task_embeddings[selected_indices]
    if any(bool(row.get("uses_eval_data")) for row in selected_views):
        raise AssertionError("official evaluation data entered Memory model-dev")

    grades = np.zeros((len(selected_views), len(memories)), dtype=np.int8)
    memory_signatures = [signature(row["signature"]) for row in memories]
    for query_index, view in enumerate(selected_views):
        task_signature = signature(view["signature"])
        for memory_index, (memory, memory_signature) in enumerate(zip(memories, memory_signatures)):
            if bool(memory["procedurally_valid"]):
                grade, _ = view_relation(view["view_type"], task_signature, memory_signature)
            else:
                grade = 0
            grades[query_index, memory_index] = int(grade)

    cosine_scores = normalized(task_embeddings) @ normalized(memory_embeddings).T
    learned_scores = {
        model_type: score_matrix(
            load_model(resolve(spec["frozen_checkpoints"][model_type]), model_type),
            task_embeddings,
            memory_embeddings,
            batch_size,
        )
        for model_type in ("gmf", "mlp", "neumf")
    }
    neumf_scores = learned_scores["neumf"]
    candidate_c = int(spec["offline_eval_contract"]["cascade_candidate_c"])
    rankings = {
        "query_cosine_full_pool": rank_descending(cosine_scores),
        "content_gmf_full_pool": rank_descending(learned_scores["gmf"]),
        "content_mlp_full_pool": rank_descending(learned_scores["mlp"]),
        "content_neumf_full_pool": rank_descending(neumf_scores),
        "cosine_then_content_neumf_c50": cosine_then_neumf(cosine_scores, neumf_scores, candidate_c),
    }
    methods = {name: memory_metrics(grades, ranking, 10) for name, ranking in rankings.items()}
    contract = spec["offline_eval_contract"]
    checks = {
        "cosine_ret1": (methods["query_cosine_full_pool"]["ret@1"], contract["cosine_ret1"]),
        "cosine_mrr": (methods["query_cosine_full_pool"]["mrr@10"], contract["cosine_mrr"]),
        "cosine_ndcg": (methods["query_cosine_full_pool"]["ndcg@10"], contract["cosine_ndcg10"]),
        "pure_ret1": (methods["content_neumf_full_pool"]["ret@1"], contract["pure_neumf_ret1"]),
        "pure_mrr": (methods["content_neumf_full_pool"]["mrr@10"], contract["pure_neumf_mrr"]),
        "pure_ndcg": (methods["content_neumf_full_pool"]["ndcg@10"], contract["pure_neumf_ndcg10"]),
        "cascade_ret1": (methods["cosine_then_content_neumf_c50"]["ret@1"], contract["cascade_ret1"]),
        "cascade_mrr": (methods["cosine_then_content_neumf_c50"]["mrr@10"], contract["cascade_mrr"]),
        "cascade_ndcg": (methods["cosine_then_content_neumf_c50"]["ndcg@10"], contract["cascade_ndcg10"]),
    }
    for label, (actual, expected) in checks.items():
        assert_close(f"memory {label}", actual, expected)

    output_rows = []
    for query_index, view in enumerate(selected_views):
        row: dict[str, Any] = {
            "query_index": query_index,
            "query_view_index": int(view["query_view_index"]),
            "query_text": view["query_text"],
            "view_type": view["view_type"],
            "grade_2_eligible": bool(np.any(grades[query_index] == 2)),
        }
        for method, ranking in rankings.items():
            order = ranking[query_index].tolist()
            row[method] = order
            row[f"{method}_top10_grades"] = grades[query_index, order[:10]].astype(int).tolist()
        output_rows.append(row)
    report = {
        "schema_version": "agent_skill_evolution.memory_offline_replay.v1",
        "evaluation_split": "train-derived internal model-dev",
        "official_dev_or_test_read": False,
        "candidate_pool": len(memories),
        "candidate_c": candidate_c,
        "final_k": 10,
        "grade_counts": {str(grade): int(np.sum(grades == grade)) for grade in (0, 1, 2)},
        "methods": methods,
    }
    write_json(output_dir / "memory_metrics.json", report)
    write_jsonl(output_dir / "memory_rankings.jsonl", output_rows)
    return {
        "report": report,
        "query_ids": [int(row["query_view_index"]) for row in selected_views],
        "candidate_ids": [int(row["memory_id"]) for row in memories],
        "gold_matrix_sha256": hashlib.sha256(grades.tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    skill = replay_skill(manifest, args.output_dir, args.batch_size)
    memory = replay_memory(manifest, args.output_dir, args.batch_size)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    lock = {
        "schema_version": "agent_skill_evolution.alfworld_ncf_evaluation_lock.v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": git_commit,
        "no_api_calls": True,
        "no_training": True,
        "skill": {
            "query_ids_sha256": sha256_json(skill["query_ids"]),
            "candidate_ids_sha256": sha256_json(skill["candidate_ids"]),
            "gold_sha256": sha256_json(skill["gold"]),
            "metrics_sha256": sha256_file(args.output_dir / "skill_metrics.json"),
            "rankings_sha256": sha256_file(args.output_dir / "skill_rankings.jsonl"),
        },
        "memory": {
            "query_ids_sha256": sha256_json(memory["query_ids"]),
            "candidate_ids_sha256": sha256_json(memory["candidate_ids"]),
            "gold_matrix_sha256": memory["gold_matrix_sha256"],
            "metrics_sha256": sha256_file(args.output_dir / "memory_metrics.json"),
            "rankings_sha256": sha256_file(args.output_dir / "memory_rankings.jsonl"),
        },
    }
    write_json(args.output_dir / "evaluation_lock.json", lock)
    summary = {
        "skill": skill["report"]["methods"],
        "memory": memory["report"]["methods"],
        "evaluation_lock": str(args.output_dir / "evaluation_lock.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
