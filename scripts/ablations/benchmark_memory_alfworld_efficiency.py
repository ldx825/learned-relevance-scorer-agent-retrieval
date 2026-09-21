#!/usr/bin/env python3
"""Benchmark Memory ALFWorld retrieval cost on one CPU thread."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/GraphOfSkills_NCF"))
from gos.ncf.models import NCFConfig, NeuMF  # noqa: E402


DEFAULT_ARRAYS = ROOT / ".runtime/memp_ncf/model_data/query_only_neumf_v17_balanced_dual_coverage/arrays.npz"
DEFAULT_MODEL = ROOT / ".runtime/memp_ncf/models/query_only_neumf_v17_balanced_dual_coverage/neumf.pt"
DEFAULT_OUTPUT = ROOT / ".runtime/ablations/alfworld_ncf_v1/efficiency/memory_alfworld/report.json"


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def worker(args: argparse.Namespace) -> int:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    with np.load(args.arrays, allow_pickle=False) as payload:
        # The latency protocol uses the 336 frozen model-dev query views.  Do
        # not move the entire training-query bank to the inference device.
        queries = torch.from_numpy(payload["task_embeddings"][:336].copy()).float().to(device)
        resources = torch.from_numpy(payload["memory_embeddings"].copy()).float().to(device)
    if len(resources) != 300:
        raise AssertionError(f"expected 300 Memory candidates, got {len(resources)}")
    queries = torch.nn.functional.normalize(queries, dim=1)
    normalized_resources = torch.nn.functional.normalize(resources, dim=1)
    model = None
    parameters = 0
    if args.method != "cosine":
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
        model = NeuMF(NCFConfig(**checkpoint["model_config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())

    @torch.inference_mode()
    def run(index: int) -> None:
        query = queries[index % len(queries)]
        if args.method == "cosine":
            torch.argsort(torch.mv(normalized_resources, query), descending=True)
        elif args.method == "full":
            assert model is not None
            scores = model(query.expand(len(resources), -1), resources)
            torch.argsort(scores, descending=True)
        else:
            assert model is not None
            cosine_scores = torch.mv(normalized_resources, query)
            candidates = torch.topk(cosine_scores, k=50, sorted=True).indices
            scores = model(query.expand(50, -1), resources[candidates])
            torch.argsort(scores, descending=True)

    for index in range(args.warmup):
        run(index)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings: list[float] = []
    for index in range(args.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        run(index)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
    report = {
        "method": args.method,
        "candidate_pool": 300,
        "candidates_scored_by_model": 0 if args.method == "cosine" else (300 if args.method == "full" else 50),
        "trainable_parameters": parameters,
        "warmup_queries": args.warmup,
        "timed_queries": args.iterations,
        "latency_ms": {
            "mean": statistics.fmean(timings),
            "median": statistics.median(timings),
            "p95": percentile(timings, 0.95),
        },
        "peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "device": str(device),
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda" else 0.0
        ),
        "gpu_used": device.type == "cuda",
        "api_calls_made": 0,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", choices=("cosine", "full", "cascade"))
    args = parser.parse_args()
    if args.method:
        return worker(args)
    methods = []
    for method in ("cosine", "full", "cascade"):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--arrays", str(args.arrays), "--model", str(args.model),
            "--warmup", str(args.warmup), "--iterations", str(args.iterations),
            "--device", args.device,
            "--method", method,
        ]
        environment = dict(os.environ)
        environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        completed = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
        methods.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    report = {
        "schema_version": "agent_skill_evolution.memory_alfworld_efficiency.v1",
        "cpu": platform.processor() or "Intel Xeon Gold 6530",
        "threads": 1,
        "device": args.device,
        "embedding_latency_included": False,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
