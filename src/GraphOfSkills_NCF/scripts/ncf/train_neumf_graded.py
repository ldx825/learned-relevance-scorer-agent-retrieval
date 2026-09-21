#!/usr/bin/env python3
"""Train the paper-style pretrained NeuMF model with 2/1/0 supervision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gos.ncf import NCFConfig
from gos.ncf.training import NeuMFGradedTrainingConfig, train_neumf_graded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arrays",
        type=Path,
        default=WORKSPACE_ROOT
        / "data/alfworld_task_skill/task_skill_v2_gos_clean/model_data/arrays.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / ".runtime/gos_ncf/models/neumf_graded_v2_clean",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    report = train_neumf_graded(
        args.arrays,
        args.output_dir,
        model_config=NCFConfig(input_dim=3072),
        training_config=NeuMFGradedTrainingConfig(
            seed=args.seed,
            batch_size=args.batch_size,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            patience=args.patience,
            target_mode="graded",
            device="cpu",
        ),
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"report: {report['outputs']['report']}")


if __name__ == "__main__":
    main()
