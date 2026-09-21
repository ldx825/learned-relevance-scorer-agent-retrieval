#!/usr/bin/env python3
"""Shared path policy for leakage-safe SkillsBench training-data scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "configs/skillbench_ncf/eval_isolation.json"


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_training_source(
    path: str | Path,
    policy: dict[str, Any],
) -> Path:
    """Resolve a path and reject anything outside the explicit source allowlist."""

    resolved = resolve_from_root(path)
    forbidden = [
        resolve_from_root(item) for item in policy["forbidden_eval_roots"]
    ]
    for root in forbidden:
        if is_within(resolved, root):
            raise PermissionError(
                f"evaluation-only path rejected by source policy: {resolved}"
            )

    allowed = [
        resolve_from_root(item) for item in policy["allowed_training_sources"]
    ]
    if not any(resolved == root or is_within(resolved, root) for root in allowed):
        raise PermissionError(
            "training source is not explicitly allowlisted: "
            f"{resolved}. Update configs/skillbench_ncf/eval_isolation.json "
            "before using a new source."
        )
    return resolved
