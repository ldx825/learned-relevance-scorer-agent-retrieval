#!/usr/bin/env python3
"""SkillDAG variant generator.

Produces tasks_skilldag/ by wrapping canonical SkillsBench tasks with the
_skilldag_template runtime (skilldag CLI + per-task skillgraph.json).

Standalone from graphskills_benchmark.py (imports shared helpers). Differences:

  * Consumes a pre-built `skillgraph.json` produced by `skilldag initialize-graph`.
    No embedding/LLM work is done inside this script.
  * Drops `skillgraph.json` into `environment/skilldag/` (bind-mounted at
    /var/lib/skilldag/runtime inside the container).
  * Patches Dockerfile to install the `skilldag` CLI wrapper pointing at the
    bind-mounted SkillDAG package source, graph file, and skill bodies.

Typical use:
  python benchmarks/skillsbench/skilldag_benchmark.py \\
    --task dialogue-parser \\
    --tasks-root data/tasks/tasks \\
    --skills-root data/skillsets/skills_200/skills_200 \\
    --skillgraph-path data/skilldag_graphs/skillgraph_200.json \\
    --skilldag-package-root . \\
    --output-root results/skillsbench_tasks/tasks_skilldag_full_200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path


SKILLSBENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SKILLSBENCH_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(SKILLSBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILLSBENCH_ROOT))

from graphskills_benchmark import (  # noqa: E402
    build_task_list,
    canonical_task_source,
    copy_task_tree,
    hardlink_or_copy_file,
    replace_task_skills,
    GRAPH_MARKER_START,
    GRAPH_MARKER_END,
    DEFAULT_TASKS_ROOT,
)

from benchmarks.shared.skilldag_prompt import full_protocol  # noqa: E402


SKILLDAG_TEMPLATE_DIR = SKILLSBENCH_ROOT / "_skilldag_template"
SKILLDAG_RETRIEVER_SKILL_DIR = REPO_ROOT / "skills" / "skilldag-retriever"
INSTRUCTION_MARKER_START = "<!-- BEGIN SKILLDAG ONLINE PROTOCOL -->"
INSTRUCTION_MARKER_END = "<!-- END SKILLDAG ONLINE PROTOCOL -->"
NCF_PLAN_MARKER_START = "<!-- BEGIN SKILLDAG NCF PROACTIVE PLAN -->"
NCF_PLAN_MARKER_END = "<!-- END SKILLDAG NCF PROACTIVE PLAN -->"
NCF_PLAN_SCHEMA = "skilldag_ncf.skillsbench_plan.v1"
NCF_BUNDLE_SCHEMA = "skilldag.portable_neumf.v1"


# ---------------------------------------------------------------------------
# Dockerfile patching — inject a skilldag-specific GRAPH block
# ---------------------------------------------------------------------------

def build_skilldag_docker_block() -> str:
    """Canonical Dockerfile snippet injected between GRAPH_MARKER_START/END.

    The generated docker-compose file bind-mounts:
      * the SkillDAG package source at /opt/skilldag/package,
      * the mutable per-task graph at /var/lib/skilldag/runtime,
      * the skill bodies at /var/lib/skilldag/bodies.

    The wrapper simply executes the package in-place. This keeps the generated
    tasks reproducible without host-side helper processes or site-specific
    network workarounds.
    """
    lines = [
        GRAPH_MARKER_START,
        # Python3 is required for the stdlib-only CLI wrapper.
        "RUN if command -v python3 >/dev/null 2>&1; then :; "
        "elif command -v apt-get >/dev/null 2>&1; then "
        "apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*; "
        'else echo "python3 is required for skilldag" >&2; exit 1; fi',
        # Use POSIX printf instead of Dockerfile heredocs. The latter produced
        # an empty wrapper under the classic builder used by rootless Docker
        # on the evaluation machine, causing every search to silently no-op.
        r"""RUN printf '%s\n' \
  '#!/usr/bin/env sh' \
  'export PYTHONPATH="/opt/skilldag/package/src:/opt/skilldag/package:${PYTHONPATH:-}"' \
  'export SKILLDAG_SKILLS_DIR="${SKILLDAG_SKILLS_DIR:-/var/lib/skilldag/bodies}"' \
  'export SKILLDAG_GRAPH_PATH="${SKILLDAG_GRAPH_PATH:-/var/lib/skilldag/runtime/skillgraph.json}"' \
  'exec python3 -m skilldag "$@"' \
  > /usr/local/bin/skilldag \
  && chmod +x /usr/local/bin/skilldag""",
        # Propagate agent instructions.
        "COPY CLAUDE.md /root/CLAUDE.md",
        "COPY AGENTS.md /root/AGENTS.md",
        "COPY GEMINI.md /root/GEMINI.md",
        'RUN for d in /app /app/workspace /app/video /root /workspace /repo; do '
        'if [ -d "$d" ]; then '
        'cp /root/CLAUDE.md "$d/CLAUDE.md" 2>/dev/null || true; '
        'cp /root/AGENTS.md "$d/AGENTS.md" 2>/dev/null || true; '
        'cp /root/GEMINI.md "$d/GEMINI.md" 2>/dev/null || true; '
        'fi; done',
        GRAPH_MARKER_END,
    ]
    return "\n".join(lines)


def patch_skilldag_dockerfile(dockerfile_path: Path) -> None:
    """Inject the skilldag block between GRAPH_MARKER_START/END.

    Order matters:
      1. Strip any upstream `COPY skills …` lines from the *original* Dockerfile
         (those copy the full skill pool into the container).
      2. Then insert our block, which does not copy any skill body or retriever
         stub into native agent skill roots.
    """
    original = dockerfile_path.read_text(encoding="utf-8")
    # Step 1: strip upstream COPY skills lines FIRST.
    stripped = re.sub(r"(?m)^\s*COPY\s+skills(?:[^\n]*)\n?", "", original)

    # Step 2: insert (or replace) our graph block.
    pattern = re.compile(
        rf"{re.escape(GRAPH_MARKER_START)}.*?{re.escape(GRAPH_MARKER_END)}\n?",
        re.DOTALL,
    )
    block = build_skilldag_docker_block() + "\n"
    if pattern.search(stripped):
        updated = pattern.sub(lambda _match: block, stripped)
    else:
        updated = stripped.rstrip() + "\n\n" + block
    dockerfile_path.write_text(updated, encoding="utf-8")


def _render_ncf_plan(plan: dict | None) -> str:
    if not plan or not plan.get("views"):
        return ""
    lines = [
        NCF_PLAN_MARKER_START,
        "NCF proactive skill plan (task-start hints; use `skilldag show <id>` before use):",
    ]
    for view in plan["views"]:
        display_query = " ".join(str(view["query"]).split())[:300]
        lines.append(f"- {view['name']} | {display_query}")
        for rank, match in enumerate(view["matches"], 1):
            description = " ".join(str(match.get("description", "")).split())[:180]
            suffix = f": {description}" if description else ""
            lines.append(f"  {rank}. {match['skill_id']}{suffix}")
    lines.append(
        "This plan does not replace online retrieval. Every later `skilldag graph search` "
        "is adaptively reranked by the same NCF model."
    )
    lines.append(NCF_PLAN_MARKER_END)
    return "\n".join(lines)


def inject_skilldag_instruction_protocol(
    instruction_path: Path,
    *,
    ncf_plan: dict | None = None,
) -> None:
    """Append a universal online SkillDAG protocol to generated task instructions.

    This is the only agent-agnostic prompt channel we can rely on across
    Terminus, ClaudeCode, Codex, and similar Harbor agents: everyone receives
    `instruction.md` as task description, while CLAUDE.md / AGENTS.md / GEMINI.md
    consumption depends on the agent runtime.
    """
    if not instruction_path.exists():
        return

    original = instruction_path.read_text(encoding="utf-8")
    proactive_plan = _render_ncf_plan(ncf_plan)
    block = (
        "\n\n"
        f"{INSTRUCTION_MARKER_START}\n"
        "Additional environment tooling for this task:\n\n"
        "- The `skilldag` CLI is available on PATH. Native agent skill roots are empty;\n"
        "  use this CLI to retrieve SkillDAG content instead of filesystem scans.\n"
        "- **MANDATORY first step before any other action**: issue\n"
        "  `skilldag graph search \"<2-5 word task summary>\" --top-k 5` to discover whether\n"
        "  a relevant skill exists. Then `skilldag show <skill_id>` to read the body of any\n"
        "  promising hit. Skipping the initial search is a protocol violation even if you\n"
        "  believe you already know how to solve the task — the graph may contain a verifier-\n"
        "  aligned authoritative interface that materially changes the right answer.\n"
        "- Retrieval is free and interruptible. Issue more searches mid-task whenever an\n"
        "  unexpected observation suggests a different skill might apply.\n\n"
        f"{proactive_plan}\n\n"
        f"{full_protocol()}\n"
        f"{INSTRUCTION_MARKER_END}\n"
    )

    pattern = re.compile(
        rf"{re.escape(INSTRUCTION_MARKER_START)}.*?{re.escape(INSTRUCTION_MARKER_END)}",
        re.DOTALL,
    )
    if pattern.search(original):
        updated = pattern.sub(block.strip(), original)
    else:
        updated = original.rstrip() + block
    instruction_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-task preparation
# ---------------------------------------------------------------------------

def collect_gold_skill_ids(source_task_dir: Path) -> list[str]:
    """Return curated SkillsBench skill ids before generated variants strip them."""
    skills_dir = source_task_dir / "environment" / "skills"
    if not skills_dir.exists():
        return []
    return sorted(
        child.name
        for child in skills_dir.iterdir()
        if child.is_dir() and (child / "SKILL.md").exists()
    )


def render_compose_for_skilldag(
    template_path: Path,
    destination_env_dir: Path,
    skills_root: Path,
    package_root: Path,
) -> str:
    """Rewrite relative mount paths in _skilldag_template/docker-compose.yaml to be
    resolvable from the rendered env dir."""
    rendered = template_path.read_text(encoding="utf-8")
    skills_relative = os.path.relpath(skills_root, destination_env_dir)
    package_relative = os.path.relpath(package_root, destination_env_dir)
    # Placeholders from template (see _skilldag_template/docker-compose.yaml):
    rendered = re.sub(
        r"(\$\{SKILLDAG_HOST_SKILLS:-)([^}]+)(\})",
        lambda m: f"{m.group(1)}{skills_relative}{m.group(3)}",
        rendered,
    )
    rendered = re.sub(
        r"(\$\{SKILLDAG_PACKAGE_ROOT:-)([^}]+)(\})",
        lambda m: f"{m.group(1)}{package_relative}{m.group(3)}",
        rendered,
    )
    return rendered


def prepare_skilldag_task(
    source_task_dir: Path,
    destination_task_dir: Path,
    skillgraph_path: Path,
    skills_root: Path,
    package_root: Path,
    *,
    ncf_bundle_path: Path | None = None,
    ncf_plan: dict | None = None,
) -> None:
    """Copy canonical task, overlay _skilldag_template files, drop workspace."""
    copy_task_tree(source_task_dir, destination_task_dir)
    replace_task_skills(destination_task_dir, SKILLDAG_RETRIEVER_SKILL_DIR)

    env_dir = destination_task_dir / "environment"
    inject_skilldag_instruction_protocol(
        destination_task_dir / "instruction.md",
        ncf_plan=ncf_plan,
    )
    for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        hardlink_or_copy_file(SKILLDAG_TEMPLATE_DIR / name, env_dir / name)
    (env_dir / "docker-compose.yaml").write_text(
        render_compose_for_skilldag(
            SKILLDAG_TEMPLATE_DIR / "docker-compose.yaml",
            destination_env_dir=env_dir,
            skills_root=skills_root,
            package_root=package_root,
        ),
        encoding="utf-8",
    )

    # Drop per-task workspace (just skillgraph.json); docker-compose mounts it
    # at /var/lib/skilldag/runtime.
    ws_dir = env_dir / "skilldag"
    if ws_dir.exists():
        shutil.rmtree(ws_dir)
    ws_dir.mkdir(parents=True, exist_ok=True)
    # skillgraph.json must be a real copy, not a hardlink: online graph edits
    # (edit-edge add/remove/retype, etc.) write back to this file, and a hardlink
    # would propagate those writes to the master skillgraph.json AND every
    # other task that shares the inode under concurrent Harbor runs.
    skillgraph_dst = ws_dir / "skillgraph.json"
    if skillgraph_dst.exists():
        skillgraph_dst.unlink()
    shutil.copy2(skillgraph_path, skillgraph_dst)
    # Co-locate the pre-warmed node-embedding cache so containers skip the
    # N-node embedding API calls at first `skilldag graph search`. Built on
    # host once via `skilldag graph search "<warmup>" --top-k 1`.
    embeddings_src = skillgraph_path.with_suffix(".embeddings.json")
    if embeddings_src.exists():
        shutil.copy2(embeddings_src, ws_dir / "skillgraph.embeddings.json")
    if ncf_bundle_path is not None:
        shutil.copy2(ncf_bundle_path, ws_dir / "ncf_bundle.json")
        (ws_dir / "ncf_plan.json").write_text(
            json.dumps(
                {
                    "schema_version": NCF_PLAN_SCHEMA,
                    "task_id": destination_task_dir.name,
                    "full_task": ncf_plan.get("full_task", "") if ncf_plan else "",
                    "views": ncf_plan["views"] if ncf_plan else [],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    # Do not copy official gold skill IDs into the container-visible runtime.
    # Host-side retrieval analysis can read the canonical task's
    # environment/skills directory after the run; the Agent must never be able
    # to inspect this evaluation-only answer file during inference.
    (ws_dir / "skills_root.txt").write_text(str(skills_root.resolve()) + "\n", encoding="utf-8")

    patch_skilldag_dockerfile(env_dir / "Dockerfile")

    # Make Harbor skip `docker compose build` for this task: when task.toml's
    # [environment] docker_image is set and YAML's `force_build: false` holds,
    # Harbor's DockerEnvironment._use_prebuilt = True and the explicit build
    # call is skipped (harbor/environments/docker/docker.py:201-215). The
    # task's own docker-compose.yaml still has a `build:` section, so a
    # missing image will fall back to auto-build via `docker compose up -d`.
    patch_task_toml_docker_image(
        destination_task_dir / "task.toml",
        image_tag=f"hb__{destination_task_dir.name}",
    )


def patch_task_toml_docker_image(toml_path: Path, image_tag: str) -> None:
    """Set `docker_image = "<image_tag>"` in task.toml's [environment] block.

    Idempotent: if the field already exists it is overwritten; if missing it
    is inserted as the last line of the [environment] block. No toml library
    dependency (the file has a simple hand-written shape).
    """
    if not toml_path.exists():
        raise FileNotFoundError(f"task.toml not found: {toml_path}")

    content = toml_path.read_text(encoding="utf-8")
    new_line = f'docker_image = "{image_tag}"'

    # Case 1: field already present — rewrite its value.
    existing = re.search(r"(?m)^docker_image\s*=.*$", content)
    if existing:
        content = content[: existing.start()] + new_line + content[existing.end():]
        toml_path.write_text(content, encoding="utf-8")
        return

    # Case 2: field absent — insert at end of [environment] block.
    # Find [environment] header then the next [section] header (or EOF).
    env_hdr = re.search(r"(?m)^\[environment\]\s*$", content)
    if not env_hdr:
        raise ValueError(f"no [environment] section in {toml_path}")
    next_hdr = re.search(r"(?m)^\[", content[env_hdr.end():])
    insert_at = env_hdr.end() + next_hdr.start() if next_hdr else len(content)

    # Trim trailing blank lines inside the block, append new_line, then restore.
    head = content[:insert_at].rstrip() + "\n"
    tail = content[insert_at:]
    content = head + new_line + "\n" + (tail if tail.startswith("\n") or not tail else "\n" + tail)
    toml_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tasks_skilldag/ from canonical tasks + SkillDAG workspace")
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT,
                        help="Canonical tasks dir (default: data/tasks/tasks)")
    parser.add_argument("--skills-root", type=Path, required=True,
                        help="Full skill pool (SKILL.md tree); mounted into container")
    parser.add_argument("--skillgraph-path", type=Path, required=True,
                        help="Path to the skillgraph.json produced by `skilldag initialize-graph`")
    parser.add_argument("--skilldag-package-root", type=Path, required=True,
                        help="SkillDAG repository root to mount into generated containers")
    parser.add_argument("--output-root", type=Path,
                        default=SKILLSBENCH_ROOT / "tasks_skilldag",
                        help="Where to emit generated variants")
    parser.add_argument("--task", action="append", default=[],
                        help="Task name (repeat for several); default: all tasks")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe output-root before generating")
    parser.add_argument(
        "--ncf-bundle",
        type=Path,
        help="Portable NeuMF JSON bundle mounted for every online search",
    )
    parser.add_argument(
        "--ncf-plan-manifest",
        type=Path,
        help="Task-start NCF plan manifest; required together with --ncf-bundle",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ncf_plan_manifest(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != NCF_PLAN_SCHEMA:
        raise ValueError(f"unsupported NCF plan manifest schema: {payload.get('schema_version')!r}")
    forbidden = {"gold", "gold_skills", "reward", "verifier", "oracle", "trajectory"}

    def contains_forbidden_key(value) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).lower() in forbidden or contains_forbidden_key(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains_forbidden_key(item) for item in value)
        return False

    if contains_forbidden_key(payload):
        raise ValueError("NCF plan manifest contains evaluation-only fields")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("NCF plan manifest must contain a non-empty tasks mapping")
    normalized: dict[str, dict] = {}
    for task_id, plan in tasks.items():
        views = plan.get("views") if isinstance(plan, dict) else None
        if not isinstance(views, list):
            raise ValueError(f"NCF plan views must be a list for task {task_id}")
        full_task = str(plan.get("full_task", "")).strip()
        if not full_task:
            raise ValueError(f"NCF plan has no full_task context for task {task_id}")
        clean_views = []
        for view in views:
            matches = view.get("matches") if isinstance(view, dict) else None
            if not view.get("name") or not view.get("query") or not isinstance(matches, list):
                raise ValueError(f"invalid NCF plan view for task {task_id}")
            if not 1 <= len(matches) <= 3:
                raise ValueError(f"NCF plan view must contain Top-1..3 matches: {task_id}")
            clean_matches = []
            for match in matches:
                skill_id = match.get("skill_id") if isinstance(match, dict) else None
                if not skill_id:
                    raise ValueError(f"NCF plan match has no skill_id: {task_id}")
                clean_matches.append(
                    {
                        "skill_id": str(skill_id),
                        "description": str(match.get("description", "")),
                    }
                )
            clean_views.append(
                {
                    "name": str(view["name"]),
                    "query": str(view["query"]),
                    "matches": clean_matches,
                }
            )
        normalized[str(task_id)] = {"full_task": full_task, "views": clean_views}
    return normalized


def main() -> None:
    args = parse_args()
    tasks_root = args.tasks_root.resolve()
    skills_root = args.skills_root.resolve()
    skillgraph_path = args.skillgraph_path.resolve()
    package_root = args.skilldag_package_root.resolve()
    output_root = args.output_root.resolve()
    ncf_bundle_path = args.ncf_bundle.resolve() if args.ncf_bundle else None
    ncf_plan_manifest_path = (
        args.ncf_plan_manifest.resolve() if args.ncf_plan_manifest else None
    )
    if bool(ncf_bundle_path) != bool(ncf_plan_manifest_path):
        raise ValueError("--ncf-bundle and --ncf-plan-manifest must be supplied together")
    ncf_plans: dict[str, dict] = {}
    if ncf_bundle_path is not None:
        if not ncf_bundle_path.is_file():
            raise FileNotFoundError(f"NCF bundle not found: {ncf_bundle_path}")
        bundle = json.loads(ncf_bundle_path.read_text(encoding="utf-8"))
        if bundle.get("schema_version") != NCF_BUNDLE_SCHEMA:
            raise ValueError(f"unsupported NCF bundle schema: {bundle.get('schema_version')!r}")
        ncf_plans = load_ncf_plan_manifest(ncf_plan_manifest_path)

    if not skillgraph_path.exists():
        raise FileNotFoundError(f"skillgraph.json not found: {skillgraph_path}")
    if not skills_root.exists():
        raise FileNotFoundError(f"skills-root not found: {skills_root}")
    if not (package_root / "src" / "skilldag").exists():
        raise FileNotFoundError(f"SkillDAG package source not found under {package_root / 'src' / 'skilldag'}")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_task_list(tasks_root, args.task)
    if not tasks:
        raise RuntimeError(f"No tasks found under {tasks_root}")

    for task_dir in tasks:
        source = canonical_task_source(task_dir)
        destination = output_root / task_dir.name
        plan = ncf_plans.get(task_dir.name) if ncf_plans else None
        if ncf_plans and plan is None:
            raise KeyError(f"NCF plan manifest has no entry for task {task_dir.name}")
        prepare_skilldag_task(
            source,
            destination,
            skillgraph_path,
            skills_root,
            package_root,
            ncf_bundle_path=ncf_bundle_path,
            ncf_plan=plan,
        )
        print(f"[skilldag] prepared {task_dir.name} → {destination}")

    if ncf_bundle_path is not None:
        (output_root / "ncf_generation_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "skilldag_ncf.skillsbench_generation.v1",
                    "tasks": len(tasks),
                    "ncf_bundle_sha256": _sha256(ncf_bundle_path),
                    "ncf_plan_manifest_sha256": _sha256(ncf_plan_manifest_path),
                    "official_gold_used_by_planner": False,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"\nGenerated {len(tasks)} variants under {output_root}")


if __name__ == "__main__":
    main()
