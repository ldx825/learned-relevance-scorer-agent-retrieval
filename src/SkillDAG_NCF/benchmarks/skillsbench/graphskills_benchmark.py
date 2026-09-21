#!/usr/bin/env python3
"""Minimal SkillsBench file helpers used by SkillDAG task generation.

This intentionally vendors only the task-copying utilities needed by
`skilldag_benchmark.py`; the original Graph-of-Skills graph construction code
is not required for SkillDAG reproduction.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path


SKILLSBENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SKILLSBENCH_ROOT.parent.parent

GRAPH_MARKER_START = "# BEGIN GRAPH SKILLS BENCHMARK"
GRAPH_MARKER_END = "# END GRAPH SKILLS BENCHMARK"
DEFAULT_TASKS_ROOT = REPO_ROOT / "data" / "tasks" / "tasks"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        hardlink_or_copy_file(item, target)


def replace_task_skills(task_dir: Path, replacement_dir: Path) -> None:
    skills_dir = task_dir / "environment" / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    link_or_copy_tree(replacement_dir, skills_dir)


def build_docker_block(_variant: str) -> str:
    lines = [
        GRAPH_MARKER_START,
        "COPY CLAUDE.md /root/CLAUDE.md",
        "COPY AGENTS.md /root/AGENTS.md",
        "COPY GEMINI.md /root/GEMINI.md",
        'RUN for d in /app /app/workspace /app/video /root /workspace /repo; do '
        'if [ -d "$d" ]; then '
        'cp /root/CLAUDE.md "$d/CLAUDE.md" 2>/dev/null || true; '
        'cp /root/AGENTS.md "$d/AGENTS.md" 2>/dev/null || true; '
        'cp /root/GEMINI.md "$d/GEMINI.md" 2>/dev/null || true; '
        "fi; done",
        GRAPH_MARKER_END,
    ]
    return "\n".join(lines)


def patch_dockerfile(dockerfile_path: Path, variant: str = "skilldag") -> None:
    original = dockerfile_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(GRAPH_MARKER_START)}.*?{re.escape(GRAPH_MARKER_END)}\n?",
        re.DOTALL,
    )
    updated = re.sub(pattern, "", original)
    updated = re.sub(r"(?m)^\s*COPY\s+skills(?:[^\n]*)\n?", "", updated)
    updated = updated.rstrip()
    block = build_docker_block(variant)
    dockerfile_path.write_text(f"{updated}\n\n{block}\n", encoding="utf-8")


def copy_task_tree(source_task_dir: Path, destination_task_dir: Path) -> None:
    if destination_task_dir.exists():
        shutil.rmtree(destination_task_dir)
    shutil.copytree(source_task_dir, destination_task_dir)


def canonical_task_source(task_dir: Path) -> Path:
    """Prefer the stable canonical task under the bundled data tree."""
    candidate = DEFAULT_TASKS_ROOT / task_dir.name
    if candidate.exists() and (candidate / "task.toml").exists():
        return candidate
    return task_dir


def build_task_list(tasks_root: Path, selected_tasks: list[str]) -> list[Path]:
    tasks = sorted(
        path
        for path in tasks_root.iterdir()
        if path.is_dir() and (path / "task.toml").exists()
    )
    if not selected_tasks:
        return tasks

    selected = set(selected_tasks)
    filtered = [path for path in tasks if path.name in selected]
    missing = selected - {path.name for path in filtered}
    if missing:
        raise FileNotFoundError(f"Unknown task ids: {', '.join(sorted(missing))}")
    return filtered
