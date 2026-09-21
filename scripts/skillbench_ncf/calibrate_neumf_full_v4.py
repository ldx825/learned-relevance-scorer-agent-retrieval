#!/usr/bin/env python3
"""Fit full-catalog train-query priors and select cosine/NCF fusion on dev."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
NCF_V3 = ROOT / "scripts/ncf_v3"
if str(NCF_V3) not in sys.path:
    sys.path.insert(0, str(NCF_V3))

from phase_neumf_common import (  # noqa: E402
    NCFConfig,
    NeuMF,
    PhaseData,
    cosine_scores,
    model_scores,
    ranking_metrics,
    utc_now,
    write_json,
)


DEFAULT_ARRAYS = ROOT / "data/skillbench_ncf/model_data/neumf_full_v4/arrays.npz"
DEFAULT_MODEL_DIR = ROOT / ".runtime/skillbench_ncf/models/neumf_full_v4"


def normalize_within_query(
    query_indices: np.ndarray, values: np.ndarray
) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for query_index in np.unique(query_indices):
        positions = np.flatnonzero(query_indices == query_index)
        current = values[positions]
        result[positions] = (current - current.mean()) / (current.std() + 1e-8)
    return result


@torch.no_grad()
def full_catalog_train_query_priors(
    model: NeuMF, data: PhaseData, *, query_batch_size: int = 16
) -> np.ndarray:
    """Average each skill's logit over unique train queries, labels unused."""
    train_query_indices = np.unique(
        data.pair_phase_indices[data.indices("train")]
    ).astype(np.int64)
    skill = data.skill_embeddings_t
    gmf_skill = model.gmf_skill_projection(skill)
    mlp_skill = model.mlp_skill_projection(skill)
    sums = np.zeros(len(data.skill_ids), dtype=np.float64)
    count = 0
    model.eval()
    for start in range(0, len(train_query_indices), query_batch_size):
        indices = train_query_indices[start : start + query_batch_size]
        query = data.phase_embeddings_t[
            torch.as_tensor(indices, dtype=torch.long, device=data.device)
        ]
        gmf_query = model.gmf_task_projection(query)
        mlp_query = model.mlp_task_projection(query)
        batch = len(query)
        gmf = gmf_query[:, None, :] * gmf_skill[None, :, :]
        left = mlp_query[:, None, :].expand(batch, len(skill), -1)
        right = mlp_skill[None, :, :].expand(batch, len(skill), -1)
        mlp = model.mlp_tower(torch.cat((left, right), dim=-1))
        logits = model.output(torch.cat((gmf, mlp), dim=-1)).squeeze(-1)
        sums += logits.sum(dim=0).cpu().numpy()
        count += batch
    if count != len(train_query_indices):
        raise AssertionError("train query prior count mismatch")
    return sums / max(count, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--public-report",
        type=Path,
        default=None,
        help="optional aggregate report without the per-skill prior vector",
    )
    parser.add_argument(
        "--report-schema-version",
        default="skillbench_ncf.online_calibration.v4",
    )
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
    )
    parser.add_argument(
        "--prior-mode",
        choices=("subtract", "none"),
        default="subtract",
        help=(
            "subtract the train-query skill prior (locked V4 behavior), or retain "
            "raw NeuMF logits when a leakage-safe audit finds little popularity bias"
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=("ndcg", "mrr"),
        default="ndcg",
        help="choose fusion alpha by internal-dev NDCG or first-relevant-rank MRR",
    )
    args = parser.parse_args()
    checkpoint = torch.load(
        args.model_dir / "neumf.pt", map_location="cpu", weights_only=True
    )
    data = PhaseData(args.arrays)
    if set(map(int, data.split_codes.tolist())) != {0, 1}:
        raise ValueError("calibration may read train/dev only")
    model = NeuMF(NCFConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Unlike a candidate-only average, this gives every one of the 1000 skills
    # a train-only generic-bias prior, including held-out-family skills.
    priors = (
        full_catalog_train_query_priors(model, data)
        if args.prior_mode == "subtract"
        else np.zeros(len(data.skill_ids), dtype=np.float64)
    )
    dev = data.indices("dev")
    dev_logits = model_scores(model, data, dev, 512)
    specific = dev_logits - priors[data.pair_skill_indices[dev]]
    query_indices = data.pair_phase_indices[dev]
    cosine_z = normalize_within_query(
        query_indices, cosine_scores(data, dev)
    )
    ncf_z = normalize_within_query(query_indices, specific)
    trials = {}
    best = None
    alpha_grid = sorted(set(args.alpha_grid))
    if not alpha_grid or any(not 0.0 <= value <= 1.0 for value in alpha_grid):
        raise ValueError("--alpha-grid values must be in [0, 1]")
    for alpha in alpha_grid:
        metrics = ranking_metrics(
            data, dev, alpha * cosine_z + (1.0 - alpha) * ncf_z
        )
        trials[str(alpha)] = metrics
        key = (
            (metrics["mrr_required"], metrics["ndcg@3"], metrics["required_recall@3"])
            if args.selection_metric == "mrr"
            else (metrics["ndcg@3"], metrics["required_recall@3"], -metrics["bad_item_rate@3"])
        )
        if best is None or key > best[0]:
            best = (key, alpha)
    if best is None:
        raise RuntimeError("no alpha selected")
    report = {
        "schema_version": args.report_schema_version,
        "generated_at": utc_now(),
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": False,
        "prior_statistics_source": (
            "all 1000 skills scored against train queries only"
            if args.prior_mode == "subtract"
            else "disabled; zero prior for every skill"
        ),
        "prior_mode": args.prior_mode,
        "alpha_selection_source": "internal family-held-out dev labeled candidates",
        "alpha_selection_metric": args.selection_metric,
        "alpha_grid": alpha_grid,
        "alpha": best[1],
        "dev_trials": trials,
        "skill_logit_prior": {
            str(skill_id): float(prior)
            for skill_id, prior in zip(data.skill_ids, priors)
        },
        "limitations": [
            "Dev labels cover the split-safe judged candidate pool, not all 1000 skills.",
            "Official SkillsBench tasks are not used for alpha or prior calibration.",
        ],
    }
    write_json(args.model_dir / "online_calibration.json", report)
    if args.public_report is not None:
        public = {key: value for key, value in report.items() if key != "skill_logit_prior"}
        args.public_report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.public_report, public)
    print(f"selected alpha={best[1]} on internal dev; official tasks read=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
