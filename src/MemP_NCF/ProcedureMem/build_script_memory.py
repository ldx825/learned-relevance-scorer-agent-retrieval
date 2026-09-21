#!/usr/bin/env python3
"""Build and embed the frozen 300-item MemP Script memory bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memp_repro import ApiConfig, CachedOpenAI, build_script_bank, ensure_bank_embeddings, sum_usage


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, default=HERE / "Alfworld/alfworld_format_traj.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-size", type=int, default=300)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--prompt-mode",
        choices=[
            "alfworld-faithful",
            "paper-minimal",
            "paper-minimal-v2",
            "paper-minimal-v3",
            "release-exact",
            "release-corrected",
        ],
        default="alfworld-faithful",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = args.output_dir / "script_bank.json"
    embedding_path = args.output_dir / "query_embeddings.json"
    api = CachedOpenAI(ApiConfig.from_env())
    bank = build_script_bank(
        trajectory_path=args.trajectories.resolve(),
        output_path=bank_path,
        api=api,
        memory_size=args.memory_size,
        workers=args.workers,
        prompt_mode=args.prompt_mode,
    )
    embeddings = ensure_bank_embeddings(bank_path, embedding_path, api)
    summary = {
        "records": len(bank["records"]),
        "unique_queries": len({item["query"] for item in bank["records"]}),
        "embedding_shape": list(embeddings.shape),
        "usage": sum_usage(item.get("usage", {}) for item in bank["records"]),
        "bank_path": str(bank_path),
        "embedding_path": str(embedding_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
