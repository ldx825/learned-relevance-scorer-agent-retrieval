#!/usr/bin/env python3
"""Audit and freeze the SkillsBench skills_1000 self-supervision sources.

This script deliberately accepts only three inputs: an extracted skill library,
its original archive, and a cold-start graph. Every input is checked against the
repository's evaluation-isolation allowlist before it is opened. It does not
read SkillsBench task instructions, gold skills, verifiers, or trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from source_policy import (
    DEFAULT_POLICY_PATH,
    ROOT,
    assert_training_source,
    load_policy,
)


DEFAULT_SKILLS_DIR = (
    ROOT / ".runtime/skilldag/data/skilldag/skillsets/skills_1000"
)
DEFAULT_ARCHIVE = ROOT / ".runtime/skilldag/downloads/skills_1000.tar.gz"
DEFAULT_GRAPH = (
    ROOT / ".runtime/skilldag/data/skilldag/skilldag_graphs/"
    "skillgraph_1000.json"
)
DEFAULT_LOCK = ROOT / "configs/skillbench_ncf/source_lock.json"
DEFAULT_OUTPUT = ROOT / "artifacts/skillbench_ncf/manifests"

IGNORED_NAMES = {".DS_Store"}
FORBIDDEN_GRAPH_KEYS = {
    "episode",
    "gold",
    "gold_skill",
    "gold_skills",
    "instruction",
    "oracle",
    "reward",
    "task_id",
    "trajectory",
    "trial",
    "verifier",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def logical_path(path: Path, source_root: Path) -> str:
    return f"skills_1000/{path.relative_to(source_root).as_posix()}"


def clean_frontmatter_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration:
        return {}

    result: dict[str, str] = {}
    for line in lines[1:end]:
        match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*?)\s*$", line)
        if match:
            result[match.group(1)] = clean_frontmatter_value(match.group(2))
    return result


def first_heading(text: str) -> str:
    in_frontmatter = False
    frontmatter_done = False
    for index, line in enumerate(text.splitlines()):
        if index == 0 and line.strip() == "---":
            in_frontmatter = True
            continue
        if in_frontmatter and line.strip() == "---":
            in_frontmatter = False
            frontmatter_done = True
            continue
        if in_frontmatter:
            continue
        if line.startswith("# "):
            return line[2:].strip()
        if frontmatter_done and line.strip():
            break
    return ""


def tree_hash(rows: list[dict[str, Any]], key: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["skill_id"])):
        digest.update(str(row["skill_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row[key]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_skill_rows(skills_dir: Path) -> list[dict[str, Any]]:
    skill_files = sorted(skills_dir.glob("*/SKILL.md"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for skill_file in skill_files:
        skill_id = skill_file.parent.name
        if skill_id in seen:
            raise ValueError(f"duplicate skill directory id: {skill_id}")
        seen.add(skill_id)
        raw = skill_file.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        frontmatter = parse_frontmatter(text)
        rows.append(
            {
                "skill_id": skill_id,
                "skill_md": logical_path(skill_file, skills_dir),
                "skill_md_sha256": sha256_bytes(raw),
                "byte_count": len(raw),
                "frontmatter_name": frontmatter.get("name", ""),
                "frontmatter_description": frontmatter.get("description", ""),
                "first_heading": first_heading(text),
            }
        )
    return rows


def build_asset_rows(skills_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(skills_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("._") or path.name in IGNORED_NAMES:
            continue
        relative = path.relative_to(skills_dir).as_posix()
        rows.append(
            {
                "relative_path": f"skills_1000/{relative}",
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
            }
        )
    return rows


def asset_tree_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["relative_path"])):
        digest.update(str(row["relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def archive_skill_hashes(archive: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not member.name.endswith("/SKILL.md"):
                continue
            parts = Path(member.name).parts
            if len(parts) != 3 or parts[0] != "skills_1000":
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            data = extracted.read()
            skill_id = parts[1]
            if skill_id in result:
                raise ValueError(f"duplicate archived skill: {skill_id}")
            result[skill_id] = sha256_bytes(data)
    return result


def find_forbidden_keys(value: Any, prefix: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            child_prefix = f"{prefix}.{key}"
            if normalized in FORBIDDEN_GRAPH_KEYS:
                hits.append(child_prefix)
            hits.extend(find_forbidden_keys(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(find_forbidden_keys(child, f"{prefix}[{index}]"))
    return hits


def require_hash(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )


def build(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    skills_dir = assert_training_source(args.skills_dir, policy)
    archive = assert_training_source(args.archive, policy)
    graph_path = assert_training_source(args.graph, policy)

    for path in (skills_dir, archive, graph_path, args.lock, args.policy):
        if not path.exists():
            raise FileNotFoundError(path)

    source_lock = json.loads(args.lock.read_text(encoding="utf-8"))
    expected = source_lock["expected"]
    archive_sha = sha256_file(archive)
    graph_sha = sha256_file(graph_path)
    require_hash(
        "skill archive",
        archive_sha,
        source_lock["sources"]["skill_archive"]["sha256"],
    )
    require_hash(
        "cold graph",
        graph_sha,
        source_lock["sources"]["cold_graph"]["sha256"],
    )

    skill_rows = build_skill_rows(skills_dir)
    if len(skill_rows) != int(expected["skill_count"]):
        raise ValueError(
            f"expected {expected['skill_count']} skills, found {len(skill_rows)}"
        )

    extracted_hashes = {
        row["skill_id"]: row["skill_md_sha256"] for row in skill_rows
    }
    archived_hashes = archive_skill_hashes(archive)
    archive_missing = sorted(set(extracted_hashes) - set(archived_hashes))
    archive_extra = sorted(set(archived_hashes) - set(extracted_hashes))
    archive_changed = sorted(
        skill_id
        for skill_id in set(extracted_hashes) & set(archived_hashes)
        if extracted_hashes[skill_id] != archived_hashes[skill_id]
    )

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    history = graph.get("history", [])
    if not isinstance(nodes, dict) or not isinstance(edges, list):
        raise ValueError("unsupported skill graph schema")

    skill_ids = set(extracted_hashes)
    node_ids = set(nodes)
    missing_graph_nodes = sorted(skill_ids - node_ids)
    extra_graph_nodes = sorted(node_ids - skill_ids)
    edge_types = Counter(str(edge.get("type", "")) for edge in edges)
    edge_origins = Counter(str(edge.get("origin", "")) for edge in edges)
    invalid_endpoints = [
        index
        for index, edge in enumerate(edges)
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids
    ]
    self_edges = [
        index
        for index, edge in enumerate(edges)
        if edge.get("source") == edge.get("target")
    ]
    edge_keys = [
        (
            str(edge.get("source")),
            str(edge.get("target")),
            str(edge.get("type")),
        )
        for edge in edges
    ]
    duplicate_edge_count = len(edge_keys) - len(set(edge_keys))
    forbidden_graph_key_paths = find_forbidden_keys(graph)

    asset_rows = build_asset_rows(skills_dir)
    skill_md_tree_sha = tree_hash(skill_rows, "skill_md_sha256")
    skill_asset_tree_sha = asset_tree_hash(asset_rows)
    checks = {
        "archive_hash_matches_lock": True,
        "graph_hash_matches_lock": True,
        "skill_count_matches_lock": len(skill_rows)
        == int(expected["skill_count"]),
        "skill_md_tree_matches_lock": skill_md_tree_sha
        == str(expected["skill_md_tree_sha256"]),
        "skill_asset_tree_matches_lock": skill_asset_tree_sha
        == str(expected["skill_asset_tree_sha256"]),
        "archive_skill_docs_match_extracted": not (
            archive_missing or archive_extra or archive_changed
        ),
        "graph_node_count_matches_lock": len(nodes)
        == int(expected["graph_node_count"]),
        "graph_edge_count_matches_lock": len(edges)
        == int(expected["graph_edge_count"]),
        "graph_nodes_match_skill_documents": not (
            missing_graph_nodes or extra_graph_nodes
        ),
        "graph_history_is_empty": len(history)
        == int(expected["graph_history_count"])
        == 0,
        "all_edges_are_cold_start": set(edge_origins)
        == {str(expected["edge_origin"])},
        "edge_types_match_lock": set(edge_types)
        == set(expected["edge_types"]),
        "all_edge_endpoints_exist": not invalid_endpoints,
        "no_self_edges": not self_edges,
        "no_duplicate_typed_edges": duplicate_edge_count == 0,
        "no_eval_signal_keys_in_graph": not forbidden_graph_key_paths,
        "all_inputs_pass_eval_isolation_allowlist": True,
    }

    audit = {
        "schema_version": "skillbench_ncf.source_audit.v1",
        "scale": source_lock["scale"],
        "source_revision": source_lock["huggingface_revision"],
        "counts": {
            "skills": len(skill_rows),
            "skill_asset_files": len(asset_rows),
            "graph_nodes": len(nodes),
            "graph_edges": len(edges),
            "graph_history": len(history),
        },
        "hashes": {
            "skill_archive_sha256": archive_sha,
            "skill_md_tree_sha256": skill_md_tree_sha,
            "skill_asset_tree_sha256": skill_asset_tree_sha,
            "cold_graph_sha256": graph_sha,
        },
        "graph": {
            "schema_version": graph.get("schema_version"),
            "updated_at": graph.get("updated_at"),
            "edge_types": dict(sorted(edge_types.items())),
            "edge_origins": dict(sorted(edge_origins.items())),
            "missing_graph_nodes": missing_graph_nodes,
            "extra_graph_nodes": extra_graph_nodes,
            "invalid_edge_indices": invalid_endpoints,
            "self_edge_indices": self_edges,
            "duplicate_typed_edge_count": duplicate_edge_count,
            "forbidden_eval_signal_key_paths": forbidden_graph_key_paths,
        },
        "archive_comparison": {
            "missing_from_archive": archive_missing,
            "extra_in_archive": archive_extra,
            "changed_after_extraction": archive_changed,
        },
        "eval_isolation": {
            "policy_schema": policy["schema_version"],
            "opened_inputs": [
                "skills_1000 extracted skill library",
                "skills_1000 source archive",
                "skillgraph_1000 cold-start graph",
                "source lock",
                "evaluation-isolation policy",
            ],
            "official_task_content_read": False,
            "note": (
                "This is an application-level path guard. The builder has no "
                "task-root input and rejects paths outside the allowlist."
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }

    output = args.output
    dump_jsonl(output / "skills_1000_manifest.jsonl", skill_rows)
    dump_json(output / "skillgraph_1000_manifest.json", {
        "schema_version": "skillbench_ncf.graph_manifest.v1",
        "scale": source_lock["scale"],
        "source_revision": source_lock["huggingface_revision"],
        "source_sha256": graph_sha,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "history_count": len(history),
        "edge_types": dict(sorted(edge_types.items())),
        "edge_origins": dict(sorted(edge_origins.items())),
    })
    dump_json(output / "source_audit_report.json", audit)
    if not audit["passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise ValueError(f"source audit failed: {', '.join(failed)}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build(args)
    print(
        "PASS: skills={skills}, nodes={nodes}, edges={edges}, history={history}".format(
            skills=audit["counts"]["skills"],
            nodes=audit["counts"]["graph_nodes"],
            edges=audit["counts"]["graph_edges"],
            history=audit["counts"]["graph_history"],
        )
    )
    print(f"skill_md_tree_sha256={audit['hashes']['skill_md_tree_sha256']}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
