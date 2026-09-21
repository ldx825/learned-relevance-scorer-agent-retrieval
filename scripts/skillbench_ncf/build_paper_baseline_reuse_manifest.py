#!/usr/bin/env python3
"""Audit reusable pure-SkillDAG trials against the saved paper task snapshot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_TASKS = ROOT / ".runtime/skillbench_ncf/official/skillsbench-v1.0/tasks"
OUTPUT = ROOT / ".runtime/skillbench_ncf/eval/paper_skilldag_once/reuse_manifest.json"
RUNS = [
    ROOT / ".runtime/skillbench_ncf/eval/baseline_smoke10/results/skillsbench-baseline-smoke10-once",
    ROOT / ".runtime/skillbench_ncf/eval/diag3_baseline_retry_results/skillsbench-diag3-baseline-retry",
]
EXCLUDED_INFRA = {
    "AgentSetupTimeoutError",
    "EnvironmentStartTimeoutError",
    "VerifierOutputParseError",
    "RewardFileNotFoundError",
    "RewardFileEmptyError",
    "VerifierTimeoutError",
    "AttributeError",
}
PROTOCOL_RE = re.compile(
    rb"\n*<!-- BEGIN SKILLDAG ONLINE PROTOCOL -->.*?"
    rb"<!-- END SKILLDAG ONLINE PROTOCOL -->\s*",
    re.DOTALL,
)


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = item.relative_to(path)
        # Finder metadata is not semantically part of a SkillsBench task.
        if relative.name == ".DS_Store" or relative.name.startswith("._"):
            continue
        digest.update(str(relative).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def generated_task_matches_canonical(generated: Path, canonical: Path) -> bool:
    """Compare every task-defining file not intentionally overlaid by SkillDAG."""
    if not generated.is_dir() or not canonical.is_dir():
        return False
    generated_instruction = PROTOCOL_RE.sub(
        b"", (generated / "instruction.md").read_bytes()
    ).strip()
    if generated_instruction != (canonical / "instruction.md").read_bytes().strip():
        return False

    def selected_files(root: Path) -> dict[str, bytes]:
        output = {}
        for item in sorted(value for value in root.rglob("*") if value.is_file()):
            relative = item.relative_to(root)
            if relative.name == ".DS_Store" or relative.name.startswith("._"):
                continue
            if relative == Path("instruction.md"):
                continue
            if relative.parts[0] == "environment" and (
                relative.name in {"Dockerfile", "docker-compose.yaml", "AGENTS.md", "CLAUDE.md", "GEMINI.md"}
                or len(relative.parts) > 1 and relative.parts[1] in {"skills", "skilldag"}
            ):
                continue
            content = item.read_bytes()
            # Harbor writes its built-image cache key back into task.toml.
            # This is runner state, not task semantics.
            if relative == Path("task.toml"):
                content = re.sub(rb"(?m)^docker_image\s*=.*\n?", b"", content)
            output[str(relative)] = content
        return output

    return selected_files(generated) == selected_files(canonical)


def reward(payload: dict) -> float | None:
    value = ((payload.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return float(value) if value is not None else None


def main() -> int:
    paper_ids = sorted(path.name for path in PAPER_TASKS.iterdir() if path.is_dir())
    if len(paper_ids) != 87:
        raise AssertionError(f"expected saved 87-task paper snapshot, got {len(paper_ids)}")
    paper_hash = {task_id: tree_hash(PAPER_TASKS / task_id) for task_id in paper_ids}

    accepted: dict[str, dict] = {}
    rejected: list[dict] = []
    for run in RUNS:
        for result_path in sorted(run.glob("*/result.json")):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            task_id = str(payload.get("task_name") or "")
            reason = None
            if task_id not in paper_hash:
                reason = "task_not_in_saved_paper_snapshot"
            generated_path = Path(((payload.get("task_id") or {}).get("path") or ""))
            if reason is None and not generated_task_matches_canonical(
                generated_path, PAPER_TASKS / task_id
            ):
                reason = "prior_generated_task_differs_from_current_target"
            agent = ((payload.get("config") or {}).get("agent") or {})
            if reason is None and agent.get("model_name") != "openai/MiniMax-M2.7":
                reason = "different_agent_model"
            if reason is None and (agent.get("kwargs") or {}).get("api_base") != "https://yunwu.ai/v1":
                reason = "different_api_route"
            exception_type = (payload.get("exception_info") or {}).get("exception_type")
            if reason is None and exception_type in EXCLUDED_INFRA:
                reason = f"excluded_infrastructure_failure:{exception_type}"
            if reason is None and reward(payload) is None and exception_type != "AgentTimeoutError":
                reason = f"no_scored_outcome:{exception_type}"
            if reason is None and task_id in accepted:
                reason = "duplicate_valid_prior_trial"
            record = {
                "task_id": task_id,
                "result_path": str(result_path.relative_to(ROOT)),
                "reward": reward(payload),
                "exception_type": exception_type,
            }
            if reason is None:
                accepted[task_id] = record
            else:
                rejected.append({**record, "reason": reason})

    remaining = sorted(set(paper_ids) - set(accepted))
    output = {
        "schema_version": "skillbench_ncf.paper_skilldag_reuse_manifest.v1",
        "paper_snapshot_tasks": len(paper_ids),
        "attempts_per_task": 1,
        "accepted_count": len(accepted),
        "remaining_count": len(remaining),
        "accepted": [accepted[key] for key in sorted(accepted)],
        "remaining_task_ids": remaining,
        "rejected": rejected,
        "compatibility_contract": {
            "target_task_set": "current upstream 87-task tree; mhc-layer-impl absent",
            "task_download_pin": None,
            "model": "openai/MiniMax-M2.7",
            "api_base": "https://yunwu.ai/v1",
            "infrastructure_failures_reused": False,
            "agent_timeout_is_scored_failure": True,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("paper_snapshot_tasks", "accepted_count", "remaining_count")}, indent=2))
    print("accepted:", ", ".join(sorted(accepted)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
