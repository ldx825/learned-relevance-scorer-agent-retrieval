#!/usr/bin/env python3
"""Fail closed when an ALFWorld NCF ablation input drifts from its lock file."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/ablations/alfworld_ncf_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_locked_files(node: Any, prefix: str = ""):
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            yield prefix.rstrip("."), node["path"], node["sha256"]
        for key, value in node.items():
            yield from iter_locked_files(value, f"{prefix}{key}.")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_locked_files(value, f"{prefix}{index}.")


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(actual - expected) <= tolerance


def check_reference_metrics(manifest: dict[str, Any], failures: list[str]) -> None:
    skill = manifest["skill_alfworld"]
    skill_report = json.loads((ROOT / skill["reference_report"]["path"]).read_text())
    skill_contract = skill["offline_eval_contract"]
    local_cosine = skill_report["overall"]["SkillDAG"]
    deployed = skill_report["overall"]["SkillDAG_NCF_V3"]
    checks = {
        "skill.query_count": (float(local_cosine["n_queries"]), float(skill_contract["query_count"])),
        "skill.local_cosine_ret1": (local_cosine["Ret@1"], skill_contract["local_cosine_ret1"]),
        "skill.local_cosine_mrr": (local_cosine["MRR"], skill_contract["local_cosine_mrr"]),
        "skill.deployed_v3_ret1": (deployed["Ret@1"], skill_contract["deployed_v3_ret1"]),
        "skill.deployed_v3_mrr": (deployed["MRR"], skill_contract["deployed_v3_mrr"]),
    }

    memory = manifest["memory_alfworld"]
    memory_report = json.loads((ROOT / memory["reference_report"]["path"]).read_text())
    memory_contract = memory["offline_eval_contract"]
    cosine = memory_report["methods"]["query_cosine"]
    pure = memory_report["methods"]["content_neumf"]
    cascade = memory_report["methods"]["cosine_then_content_neumf"]
    checks.update(
        {
            "memory.query_count": (float(memory_report["query_count"]), float(memory_contract["query_count"])),
            "memory.memory_count": (float(memory_report["memory_count"]), float(memory_contract["candidate_pool"])),
            "memory.cosine_ret1": (cosine["ret@1"], memory_contract["cosine_ret1"]),
            "memory.cosine_mrr": (cosine["mrr@10"], memory_contract["cosine_mrr"]),
            "memory.cosine_ndcg10": (cosine["ndcg@10"], memory_contract["cosine_ndcg10"]),
            "memory.pure_neumf_ret1": (pure["ret@1"], memory_contract["pure_neumf_ret1"]),
            "memory.pure_neumf_mrr": (pure["mrr@10"], memory_contract["pure_neumf_mrr"]),
            "memory.pure_neumf_ndcg10": (pure["ndcg@10"], memory_contract["pure_neumf_ndcg10"]),
            "memory.cascade_ret1": (cascade["ret@1"], memory_contract["cascade_ret1"]),
            "memory.cascade_mrr": (cascade["mrr@10"], memory_contract["cascade_mrr"]),
            "memory.cascade_ndcg10": (cascade["ndcg@10"], memory_contract["cascade_ndcg10"]),
        }
    )
    for name, (actual, expected) in checks.items():
        if not close(float(actual), float(expected)):
            failures.append(f"metric mismatch {name}: actual={actual!r} expected={expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-git-commit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    files: list[dict[str, Any]] = []
    for label, raw_path, expected in iter_locked_files(manifest):
        path = ROOT / raw_path
        row: dict[str, Any] = {"label": label, "path": str(path), "expected_sha256": expected}
        if not path.is_file():
            row["status"] = "missing"
            failures.append(f"missing {label}: {path}")
        else:
            actual = sha256(path)
            row["actual_sha256"] = actual
            row["status"] = "ok" if actual == expected else "mismatch"
            if actual != expected:
                failures.append(f"hash mismatch {label}: {path}")
        files.append(row)

    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    expected_commit = manifest["created_from_git_commit"]
    if args.require_git_commit and current_commit != expected_commit:
        failures.append(f"git commit mismatch: actual={current_commit} expected={expected_commit}")

    if not failures:
        check_reference_metrics(manifest, failures)
    report = {
        "schema_version": "agent_skill_evolution.ncf_ablation_preflight.v1",
        "manifest": str(manifest_path),
        "git_commit": current_commit,
        "locked_git_commit": expected_commit,
        "git_commit_enforced": args.require_git_commit,
        "files": files,
        "failures": failures,
        "ok": not failures,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
