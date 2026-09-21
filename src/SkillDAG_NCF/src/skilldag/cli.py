#!/usr/bin/env python3
"""skillDAG CLI — search, view, retrieve, and operate on SkillGraph state."""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .graph import DEFAULT_GRAPH_PATH, DEFAULT_SEARCH_DEPTH, GraphCLIError, SkillGraph, SYMMETRIC_EDGE_TYPES
from .search import DEFAULT_SKILLS_DIR

CLI_DEFAULT_SKILLS_DIR = Path(os.environ.get("SKILLDAG_SKILLS_DIR", str(DEFAULT_SKILLS_DIR)))
CLI_DEFAULT_GRAPH_PATH = Path(os.environ.get("SKILLDAG_GRAPH_PATH", str(DEFAULT_GRAPH_PATH)))


def _emit_json(payload: dict, exit_code: int = 0):
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(exit_code)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _build_graph_context(skill_id: str, graph_path: Path) -> str:
    """Return a compact graph-context block for skill_id, or '' if no edges / graph absent."""
    if not graph_path.exists():
        return ""
    try:
        raw = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if skill_id not in raw.get("nodes", {}):
        return ""

    by_type: dict[str, list[str]] = {}
    for edge in raw.get("edges", []):
        etype = edge["type"]
        src, tgt = edge["source"], edge["target"]
        if src == skill_id:
            by_type.setdefault(etype, []).append(tgt)
        elif tgt == skill_id and etype in SYMMETRIC_EDGE_TYPES:
            by_type.setdefault(etype, []).append(src)

    if not by_type:
        return ""

    lines = ["## Graph context"]
    for etype in ("depends_on", "composes_with", "similar_to", "conflicts_with", "specializes"):
        targets = by_type.get(etype)
        if targets:
            lines.append(f"{etype}: {', '.join(sorted(targets))}")
    return "\n".join(lines) + "\n"


def cmd_show(args):
    skills_dir = Path(args.skills_dir)
    skill_ids = args.skill_id if isinstance(args.skill_id, list) else [args.skill_id]
    multi = len(skill_ids) > 1

    available = None
    parts: list[str] = []
    missing: list[str] = []

    for sid in skill_ids:
        skill_path = skills_dir / sid / "SKILL.md"
        if not skill_path.exists():
            missing.append(sid)
            continue
        graph_ctx = _build_graph_context(sid, Path(args.graph_path))
        content = skill_path.read_text()
        body = f"{graph_ctx}\n{content}" if graph_ctx else content
        if multi:
            parts.append(f"=== {sid} ===\n{body}")
        else:
            parts.append(body)

    for sid in missing:
        print(f"Error: skill '{sid}' not found in {skills_dir}.", file=sys.stderr)
        if available is None:
            available = (
                [
                    d.name
                    for d in skills_dir.iterdir()
                    if d.is_dir() and (d / "SKILL.md").exists()
                ]
                if skills_dir.exists()
                else []
            )
        candidates = difflib.get_close_matches(sid, available, n=5, cutoff=0.4)
        if candidates:
            print(f"  Did you mean: {', '.join(candidates)}?", file=sys.stderr)

    if not parts:
        sys.exit(1)

    output = "\n\n".join(parts)
    if args.pager and sys.stdout.isatty():
        pager = shutil.which("less") or shutil.which("more") or None
        if pager:
            subprocess.run([pager, "-R"], input=output, text=True)
            return

    print(output)
    if missing:
        sys.exit(2)


def cmd_add(args):
    skills_dir = Path(args.skills_dir)
    dest = Path(args.destination)

    copied, skipped = 0, 0
    for skill_id in args.skill_ids:
        src = skills_dir / skill_id
        if not src.exists():
            print(f"  skip: {skill_id} (not found in {skills_dir})", file=sys.stderr)
            skipped += 1
            continue

        dst = dest / skill_id
        if dst.exists() and not args.overwrite:
            print(f"  skip: {skill_id} (already exists at {dst}, use --overwrite)")
            skipped += 1
            continue

        if args.dry_run:
            file_count = sum(1 for _ in src.rglob("*") if _.is_file())
            print(f"  [dry-run] would copy {src} -> {dst} ({file_count} files)")
        else:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            file_count = sum(1 for _ in dst.rglob("*") if _.is_file())
            print(f"  added: {skill_id} ({file_count} files)")
        copied += 1

    action = "would copy" if args.dry_run else "copied"
    print(f"\n  {action}: {copied}, skipped: {skipped}")


def _graph_context(args) -> SkillGraph:
    return SkillGraph.load(graph_path=args.graph_path, skills_dir=args.skills_dir)


def cmd_graph_search(args):
    graph = _graph_context(args)
    queries = args.query if isinstance(args.query, list) else [args.query]

    # Single-query path: preserve original schema for backward compat.
    if len(queries) == 1:
        q = queries[0]
        result = graph.search(q, top_k=args.top_k, depth=args.depth)
        _emit_json({
            "ok": True,
            "op": "search",
            "result": {
                "query": q,
                "top_k": args.top_k,
                "depth": args.depth,
                "matches": result["matches"],
                "neighbors": result["neighbors"],
                "conflicts": result["conflicts"],
            },
        })
        return

    # Batch path: ONE embedding API call covers all N queries.
    batch = graph.search_batch(queries, top_k=args.top_k, depth=args.depth)
    _emit_json({
        "ok": True,
        "op": "search",
        "result": {
            "batch": True,
            "top_k": args.top_k,
            "depth": args.depth,
            "queries": queries,
            "results": [
                {
                    "query": q,
                    "matches": r["matches"],
                    "neighbors": r["neighbors"],
                    "conflicts": r["conflicts"],
                }
                for q, r in zip(queries, batch)
            ],
        },
    })


def cmd_graph_get_skill(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "get-skill",
            "result": graph.get_skill(args.skill_id),
        }
    )


def cmd_graph_get_dependencies(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "get-dependencies",
            "result": graph.get_dependencies(args.skill_id, transitive=args.transitive),
        }
    )


def cmd_graph_get_conflicts(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "get-conflicts",
            "result": graph.get_conflicts(args.skill_id),
        }
    )


def cmd_graph_get_alternatives(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "get-alternatives",
            "result": graph.get_alternatives(args.skill_id),
        }
    )


def cmd_graph_check_set(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "check-set",
            "result": graph.check_set(args.skill_ids),
        }
    )


def cmd_graph_expand_set(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "expand-set",
            "result": graph.expand_set(args.skill_ids),
        }
    )


def cmd_graph_repair_set(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "repair-set",
            "result": graph.repair_set(args.skill_ids),
        }
    )


# ---------------------------------------------------------------------------
# Propose (dry-run) — returns {would, related}, never mutates the graph.
# ---------------------------------------------------------------------------

def cmd_graph_propose_edge(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "propose-edge",
            "result": graph.propose_edge(
                args.source, args.target, args.edge_type, args.reason
            ),
        }
    )


def cmd_graph_propose_remove(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "propose-remove",
            "result": graph.propose_remove_edge(
                args.source, args.target, args.edge_type, args.reason
            ),
        }
    )


def cmd_graph_propose_retype(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "propose-retype",
            "result": graph.propose_retype_edge(
                args.source, args.target, args.from_type, args.to_type, args.reason
            ),
        }
    )


# ---------------------------------------------------------------------------
# Edit (commit) — only mutation path. Add / remove / retype.
# ---------------------------------------------------------------------------

def cmd_graph_edit_add(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "edit-edge",
            "result": graph.edit_edge(
                "add",
                source=args.source,
                target=args.target,
                edge_type=args.edge_type,
                reason=args.reason,
                task_id=args.task_id,
            ),
        }
    )


def cmd_graph_edit_remove(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "edit-edge",
            "result": graph.edit_edge(
                "remove",
                source=args.source,
                target=args.target,
                edge_type=args.edge_type,
                reason=args.reason,
                task_id=args.task_id,
            ),
        }
    )


def cmd_graph_edit_retype(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "edit-edge",
            "result": graph.edit_edge(
                "retype",
                source=args.source,
                target=args.target,
                from_type=args.from_type,
                to_type=args.to_type,
                reason=args.reason,
                task_id=args.task_id,
            ),
        }
    )


def cmd_graph_edit_rollback(args):
    graph = _graph_context(args)
    _emit_json(
        {
            "ok": True,
            "op": "edit-rollback",
            "result": graph.rollback(
                steps=args.steps,
                task_id=args.task_id,
                reason=args.reason,
            ),
        }
    )


def cmd_initialize_graph(args):
    graph_path = Path(args.graph_path)
    skills_dir = Path(args.skills_dir)

    if graph_path.exists():
        if not args.force:
            _emit_json(
                {
                    "ok": False,
                    "error": {
                        "code": "GRAPH_EXISTS",
                        "message": f"{graph_path} already exists. Use --force to overwrite.",
                    },
                },
                exit_code=2,
            )
        graph_path.unlink()

    os.environ["SKILLDAG_INITIALIZE_ENABLED"] = "1"

    graph = SkillGraph.load(graph_path=graph_path, skills_dir=skills_dir)
    edge_counts: dict[str, int] = {}
    for e in graph.data.get("edges", []):
        edge_counts[e["type"]] = edge_counts.get(e["type"], 0) + 1

    _emit_json(
        {
            "ok": True,
            "op": "initialize-graph",
            "result": {
                "path": str(graph_path),
                "nodes": len(graph.data.get("nodes", {})),
                "edges": len(graph.data.get("edges", [])),
                "edges_by_type": edge_counts,
            },
        }
    )


def cmd_help(args):
    print("""\
skilldag — skill graph CLI

SKILL COMMANDS
  skilldag show <skill_id>             Print full SKILL.md content (use to load a skill)
  skilldag add <skill_id>... --to <dir> Copy skill directories to target path

GRAPH COMMANDS  (all return JSON)
  skilldag graph search <query>        Vector top-K matches (cosine on
                                       text-embedding-3-large) + neighbor list
                                         (opts: --top-k K [cosine matches],
                                                --depth D [0=no neighbors, 1=default, 2+=deeper])
  skilldag graph get-skill <id>        Get node metadata
  skilldag graph get-dependencies <id> List skills this one depends on
  skilldag graph get-conflicts <id>    List conflicting skills
  skilldag graph get-alternatives <id> List similar/specialized alternatives
  skilldag graph check-set <id>...     Validate a skill set (deps + conflicts)
  skilldag graph expand-set <id>...    Expand set with transitive dependencies
  skilldag graph repair-set <id>...    Suggest repairs for a conflicting set

GRAPH MUTATION  (two-step: propose [dry-run] → edit-edge [commit])

  Propose (pure dry-run; returns {would, related} with prior reasons, never mutates).
  --reason is required as a forcing function — articulate the evidence before commit:
    skilldag graph propose-edge    <src> <tgt> <type>                --reason <r>
    skilldag graph propose-remove  <src> <tgt> <type>                --reason <r>
    skilldag graph propose-retype  <src> <tgt> --from <t1> --to <t2> --reason <r>

  Edit (commit the change immediately; --reason required, --task-id optional):
    skilldag graph edit-edge add    <src> <tgt> <type> --reason <r>
    skilldag graph edit-edge remove <src> <tgt> <type> --reason <r>
    skilldag graph edit-edge retype <src> <tgt> --from <t1> --to <t2> --reason <r>

  Valid edge types: depends_on | composes_with | similar_to | conflicts_with | specializes
""")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="skilldag",
        description="Search and retrieve skills from the 34k skill pool.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- show ---
    p_show = sub.add_parser("show", help="Show full SKILL.md for one or more skills")
    p_show.add_argument("skill_id", nargs="+",
                        help="One or more skill IDs (e.g. owner--skill-name); when multiple are given, each body is prefixed with `=== <id> ===`")
    p_show.add_argument("--pager", action="store_true", help="Use pager for output")
    p_show.add_argument("--skills-dir", type=str, default=str(CLI_DEFAULT_SKILLS_DIR),
                        help="Path to skills directory")
    p_show.add_argument("--graph-path", type=str, default=str(CLI_DEFAULT_GRAPH_PATH),
                        help="Path to graph JSON (for graph context header)")
    p_show.set_defaults(func=cmd_show)

    # --- add ---
    p_add = sub.add_parser("add", help="Copy skills to a target directory")
    p_add.add_argument("skill_ids", nargs="+", metavar="skill_id",
                        help="One or more skill IDs to copy")
    p_add.add_argument("--to", type=str, required=True, dest="destination",
                        help="Target directory")
    p_add.add_argument("--overwrite", action="store_true", help="Overwrite existing skills")
    p_add.add_argument("--dry-run", action="store_true", help="Preview without copying")
    p_add.add_argument("--skills-dir", type=str, default=str(CLI_DEFAULT_SKILLS_DIR),
                        help="Path to skills directory")
    p_add.set_defaults(func=cmd_add)

    # --- help ---
    p_help = sub.add_parser("help", help="Show a concise command reference")
    p_help.set_defaults(func=cmd_help)

    # --- graph ---
    p_graph = sub.add_parser("graph", help="Operate on the single SkillGraph JSON")
    p_graph.add_argument("--graph-path", type=str, default=str(CLI_DEFAULT_GRAPH_PATH),
                         help="Path to the full graph JSON")
    p_graph.add_argument("--skills-dir", type=str, default=str(CLI_DEFAULT_SKILLS_DIR),
                         help="Path to skills directory (used to sync nodes)")
    graph_sub = p_graph.add_subparsers(dest="graph_command", required=True)

    # Alias: `skilldag graph show <id> [<id> ...]` == `skilldag show <id> [<id> ...]`.
    p_gshow = graph_sub.add_parser("show", help="Show full SKILL.md for one or more skills (alias of `skilldag show`)")
    p_gshow.add_argument("skill_id", nargs="+", help="One or more skill IDs")
    p_gshow.add_argument("--pager", action="store_true", help="Use pager for output")
    p_gshow.set_defaults(func=cmd_show)

    p_gsearch = graph_sub.add_parser(
        "search",
        help="Vector top-K matches (cosine on text-embedding-3-large) + depth-bounded neighbor list.",
    )
    p_gsearch.add_argument(
        "query",
        nargs="+",
        help=(
            "Task query (one or more). Single query uses original output schema "
            "(matches/neighbors/conflicts). Multiple queries use batch schema "
            "({queries: [...], results: [{query, matches, ...}, ...]}) and run "
            "in ONE embedding API call."
        ),
    )
    p_gsearch.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max cosine matches returned (default: 5). Graph neighbors do NOT compete for these slots.",
    )
    p_gsearch.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_SEARCH_DEPTH,
        help=(
            f"BFS depth for neighbor list around the cosine matches "
            f"(default: {DEFAULT_SEARCH_DEPTH}). 0 disables neighbors entirely. "
            f"Edge types eligible for walking are controlled by "
            f"WALKABLE_EDGE_TYPES (currently similar_to, specializes, "
            f"composes_with, depends_on). conflicts_with edges are never traversed."
        ),
    )
    p_gsearch.set_defaults(func=cmd_graph_search)

    p_gskill = graph_sub.add_parser("get-skill", help="Get a node summary")
    p_gskill.add_argument("skill_id", help="Skill id")
    p_gskill.set_defaults(func=cmd_graph_get_skill)

    p_gdeps = graph_sub.add_parser("get-dependencies", help="Get skill dependencies")
    p_gdeps.add_argument("skill_id", help="Skill id")
    p_gdeps.add_argument("--transitive", action="store_true", help="Return transitive dependencies")
    p_gdeps.set_defaults(func=cmd_graph_get_dependencies)

    p_gconf = graph_sub.add_parser("get-conflicts", help="Get skill conflicts")
    p_gconf.add_argument("skill_id", help="Skill id")
    p_gconf.set_defaults(func=cmd_graph_get_conflicts)

    p_galts = graph_sub.add_parser("get-alternatives", help="Get similar/specialized alternatives")
    p_galts.add_argument("skill_id", help="Skill id")
    p_galts.set_defaults(func=cmd_graph_get_alternatives)

    p_gcheck = graph_sub.add_parser("check-set", help="Check a skill set for missing deps/conflicts/redundancy")
    p_gcheck.add_argument("skill_ids", nargs="+", help="Skill ids")
    p_gcheck.set_defaults(func=cmd_graph_check_set)

    p_gexpand = graph_sub.add_parser("expand-set", help="Expand a skill set with transitive dependencies")
    p_gexpand.add_argument("skill_ids", nargs="+", help="Skill ids")
    p_gexpand.set_defaults(func=cmd_graph_expand_set)

    p_grepair = graph_sub.add_parser("repair-set", help="Suggest how to repair a problematic skill set")
    p_grepair.add_argument("skill_ids", nargs="+", help="Skill ids")
    p_grepair.set_defaults(func=cmd_graph_repair_set)

    # --- propose (dry-run) ---
    p_gprop = graph_sub.add_parser(
        "propose-edge",
        help="Dry-run add: return {would, related} with prior reasons for this pair; never mutates.",
    )
    p_gprop.add_argument("source", help="Source skill id")
    p_gprop.add_argument("target", help="Target skill id")
    p_gprop.add_argument("edge_type", help="Edge type")
    p_gprop.add_argument("--reason", required=True, help="Reason (required, <=200 chars)")
    p_gprop.set_defaults(func=cmd_graph_propose_edge)

    p_gprem = graph_sub.add_parser(
        "propose-remove",
        help="Dry-run remove: return {would, related} with prior reasons for this pair; never mutates.",
    )
    p_gprem.add_argument("source", help="Source skill id")
    p_gprem.add_argument("target", help="Target skill id")
    p_gprem.add_argument("edge_type", help="Edge type to remove")
    p_gprem.add_argument("--reason", required=True, help="Reason (required, <=200 chars)")
    p_gprem.set_defaults(func=cmd_graph_propose_remove)

    p_gpret = graph_sub.add_parser(
        "propose-retype",
        help="Dry-run retype: return {would, related} with prior reasons for this pair; never mutates.",
    )
    p_gpret.add_argument("source", help="Source skill id")
    p_gpret.add_argument("target", help="Target skill id")
    p_gpret.add_argument("--from", dest="from_type", required=True, help="Existing edge type")
    p_gpret.add_argument("--to", dest="to_type", required=True, help="New edge type")
    p_gpret.add_argument("--reason", required=True, help="Reason (required, <=200 chars)")
    p_gpret.set_defaults(func=cmd_graph_propose_retype)

    # --- edit-edge (commit) ---
    p_gedit = graph_sub.add_parser(
        "edit-edge",
        help="Commit a graph change: add / remove / retype. Requires --reason and --task-id.",
    )
    edit_sub = p_gedit.add_subparsers(dest="edit_action", required=True)

    p_eadd = edit_sub.add_parser("add", help="Add an edge immediately")
    p_eadd.add_argument("source", help="Source skill id")
    p_eadd.add_argument("target", help="Target skill id")
    p_eadd.add_argument("edge_type", help="Edge type")
    p_eadd.add_argument("--reason", required=True, help="Reason (required, <=200 chars stored)")
    p_eadd.add_argument("--task-id", default="", help="Task id (optional; recorded in evidence if given)")
    p_eadd.set_defaults(func=cmd_graph_edit_add)

    p_erem = edit_sub.add_parser("remove", help="Remove an existing edge immediately")
    p_erem.add_argument("source", help="Source skill id")
    p_erem.add_argument("target", help="Target skill id")
    p_erem.add_argument("edge_type", help="Edge type to remove")
    p_erem.add_argument("--reason", required=True, help="Reason (required)")
    p_erem.add_argument("--task-id", default="", help="Task id (optional; recorded in evidence if given)")
    p_erem.set_defaults(func=cmd_graph_edit_remove)

    p_eret = edit_sub.add_parser("retype", help="Retype an existing edge immediately")
    p_eret.add_argument("source", help="Source skill id")
    p_eret.add_argument("target", help="Target skill id")
    p_eret.add_argument("--from", dest="from_type", required=True, help="Existing edge type")
    p_eret.add_argument("--to", dest="to_type", required=True, help="New edge type")
    p_eret.add_argument("--reason", required=True, help="Reason (required)")
    p_eret.add_argument("--task-id", default="", help="Task id (optional; recorded in evidence if given)")
    p_eret.set_defaults(func=cmd_graph_edit_retype)

    p_erb = edit_sub.add_parser(
        "rollback",
        help="Revert committed edit(s): last --steps N (LIFO), or all edits for --task-id. Appends an auditable rollback_edge entry.",
    )
    p_erb.add_argument("--steps", type=int, default=1, help="Number of most-recent reversible edits to undo (ignored if --task-id given)")
    p_erb.add_argument("--task-id", default="", help="Revert every committed edit whose evidence references this task id")
    p_erb.add_argument("--reason", required=True, help="Reason (required, <=200 chars stored)")
    p_erb.set_defaults(func=cmd_graph_edit_rollback)

    # --- initialize-graph ---
    p_init = sub.add_parser(
        "initialize-graph",
        help="Build skillgraph.json from SKILL.md files via embedding + LLM pair classification "
             "(build-time only; requires SKILLDAG_EMBEDDING_API_KEY and SKILLDAG_LLM_API_KEY)",
    )
    p_init.add_argument("--graph-path", type=str, default=str(CLI_DEFAULT_GRAPH_PATH),
                        help="Output graph JSON path")
    p_init.add_argument("--skills-dir", type=str, default=str(CLI_DEFAULT_SKILLS_DIR),
                        help="Path to skills directory to scan for nodes")
    p_init.add_argument("--force", action="store_true",
                        help="Overwrite existing skillgraph.json")
    p_init.set_defaults(func=cmd_initialize_graph)

    args = parser.parse_args()
    try:
        args.func(args)
    except GraphCLIError as exc:
        _emit_json(exc.to_dict(), exit_code=1)


if __name__ == "__main__":
    main()
