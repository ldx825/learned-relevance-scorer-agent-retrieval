#!/usr/bin/env python3
"""Train SkillsBench V4 with the ALFWorld V3 pointwise+pairwise NeuMF recipe."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
NCF_V3 = ROOT / "scripts/ncf_v3"
if str(NCF_V3) not in sys.path:
    sys.path.insert(0, str(NCF_V3))

from phase_neumf_common import (  # noqa: E402
    GMF,
    MLP,
    NeuMF,
    NCFConfig,
    PhaseData,
    TrainConfig,
    checkpoint_payload,
    evaluate_cosine,
    evaluate_model,
    evaluate_popularity,
    preference_pairs,
    seed_all,
    train_one,
    train_popularity,
    utc_now,
    write_json,
)


DEFAULT_ARRAYS = ROOT / "data/skillbench_ncf/model_data/neumf_full_v4/arrays.npz"
DEFAULT_OUTPUT = ROOT / ".runtime/skillbench_ncf/models/neumf_full_v4"
DEFAULT_PUBLIC_REPORT = (
    ROOT / "artifacts/skillbench_ncf/manifests/neumf_full_v4_training_report.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--public-report", type=Path, default=DEFAULT_PUBLIC_REPORT)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pair-batch-size", type=int, default=512)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--sgd-lr", type=float, default=1e-2)
    parser.add_argument("--pretrain-alpha", type=float, default=0.5)
    parser.add_argument("--pairwise-lambda", type=float, default=0.20)
    parser.add_argument(
        "--selection-metric",
        choices=("ndcg", "mrr", "final"),
        default="ndcg",
        help="internal-dev checkpoint criterion (default preserves historical NDCG selection)",
    )
    parser.add_argument(
        "--report-schema-version",
        default="skillbench_ncf.neumf_full_training.v4",
        help="report schema identifier; lets later leakage-safe corpora reuse the locked trainer",
    )
    parser.add_argument(
        "--experiment-label",
        default="SkillsBench V4",
        help="human-readable dataset label used in validation errors and reports",
    )
    parser.add_argument(
        "--official-skillsbench-tasks-used",
        action="store_true",
        help="declare supervised use of the frozen official development split",
    )
    args = parser.parse_args()

    config = TrainConfig(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        pair_batch_size=args.pair_batch_size,
        adam_lr=args.adam_lr,
        sgd_lr=args.sgd_lr,
        alpha=args.pretrain_alpha,
        pairwise_lambda=args.pairwise_lambda,
    )
    seed_all(config.seed)
    if (args.output / "training_report.json").exists():
        raise FileExistsError(
            "output already contains a completed training report; use a new directory"
        )
    data = PhaseData(args.arrays, config.device)
    split_counts = Counter(map(int, data.split_codes.tolist()))
    expected_splits = {0} if args.selection_metric == "final" else {0, 1}
    if set(split_counts) != expected_splits:
        raise ValueError(f"{args.experiment_label} requires splits {expected_splits}, got {split_counts}")
    if len(data.skill_ids) != 1_000 or data.input_dim != 3_072:
        raise ValueError("unexpected SkillsBench catalog or embedding dimension")

    args.output.mkdir(parents=True, exist_ok=True)
    model_config = NCFConfig(
        input_dim=data.input_dim,
        gmf_latent_dim=16,
        mlp_layer_sizes=(64, 32, 16),
    )
    preferences = preference_pairs(data)
    print(
        f"train pairs={len(data.indices('train'))} "
        f"preference pairs={len(preferences[0])}",
        flush=True,
    )

    gmf = GMF(model_config)
    gmf_info = train_one(
        gmf,
        data,
        torch.optim.Adam(gmf.parameters(), lr=config.adam_lr),
        config,
        config.pretrain_epochs,
        config.seed + 1,
        preferences,
        args.selection_metric,
    )
    mlp = MLP(model_config)
    mlp_info = train_one(
        mlp,
        data,
        torch.optim.Adam(mlp.parameters(), lr=config.adam_lr),
        config,
        config.pretrain_epochs,
        config.seed + 2,
        preferences,
        args.selection_metric,
    )
    neumf = NeuMF(model_config)
    neumf.load_pretrained(gmf, mlp, alpha=config.alpha)
    neumf_info = train_one(
        neumf,
        data,
        torch.optim.SGD(neumf.parameters(), lr=config.sgd_lr),
        config,
        config.finetune_epochs,
        config.seed + 3,
        preferences,
        args.selection_metric,
    )
    torch.save(checkpoint_payload(gmf, model_config, config), args.output / "gmf.pt")
    torch.save(checkpoint_payload(mlp, model_config, config), args.output / "mlp.pt")
    torch.save(checkpoint_payload(neumf, model_config, config), args.output / "neumf.pt")

    popularity = train_popularity(data)
    report = {
        "schema_version": args.report_schema_version,
        "experiment_label": args.experiment_label,
        "generated_at": utc_now(),
        "api_calls_made": 0,
        "official_skillsbench_tasks_used": args.official_skillsbench_tasks_used,
        "initialization": "from_scratch",
        "losses": {
            "pointwise": "weighted BCE with target=grade/2",
            "pairwise": "within-query 2>1, 2>0, 1>0 softplus ranking",
            "pairwise_lambda": config.pairwise_lambda,
        },
        "checkpoint_selection_metric": args.selection_metric,
        "model_config": asdict(model_config),
        "train_config": asdict(config),
        "counts": {
            "train_pairs": len(data.indices("train")),
            "dev_pairs": len(data.indices("dev")),
            "preference_pairs": len(preferences[0]),
        },
        "training": {"gmf": gmf_info, "mlp": mlp_info, "neumf": neumf_info},
        "dev_metrics": (
            None if args.selection_metric == "final" else {
                "cosine": evaluate_cosine(data, "dev"),
                "train_only_skill_popularity": evaluate_popularity(data, "dev", popularity),
                "neumf": evaluate_model(neumf, data, "dev", config.batch_size),
            }
        ),
        "checkpoint": str(args.output / "neumf.pt"),
    }
    write_json(args.output / "training_report.json", report)
    public = {key: value for key, value in report.items() if key != "training"}
    public["training_summary"] = {
        name: {
            "best_epoch": value["best_epoch"],
            "best_dev_selection_key": value["best_dev_selection_key"],
            "epochs_ran": len(value["history"]),
        }
        for name, value in report["training"].items()
    }
    args.public_report.parent.mkdir(parents=True, exist_ok=True)
    args.public_report.write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["dev_metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
