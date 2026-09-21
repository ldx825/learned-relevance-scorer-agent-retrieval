"""Extract versioned ALFWorld tasks and SkillDAG skills without API calls."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "valid_train", "valid_seen", "valid_unseen")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text.strip()
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text.strip()
    metadata: dict[str, str] = {}
    for line in text[4:end].splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, text[end + 4 :].lstrip("\r\n").strip()


def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _task_record(path: Path, root: Path, split: str) -> dict[str, Any]:
    raw = path.read_bytes()
    source = json.loads(raw)
    annotations = []
    for annotation in source.get("turk_annotations", {}).get("anns", []) or []:
        annotations.append(
            {
                "task_desc": str(annotation.get("task_desc", "")).strip(),
                "high_descs": _unique_strings(annotation.get("high_descs", []) or []),
                "votes": annotation.get("votes", []) or [],
            }
        )

    task_text_variants = _unique_strings(item["task_desc"] for item in annotations)
    high_plan = []
    for step in source.get("plan", {}).get("high_pddl", []) or []:
        action = step.get("discrete_action", {}) or {}
        high_plan.append(
            {
                "action": str(action.get("action", "")).strip(),
                "args": action.get("args", []) or [],
            }
        )

    relative_path = path.relative_to(root).as_posix()
    record_id = f"alfworld:{split}:{path.parent.relative_to(root / split).as_posix()}"
    primary_text = task_text_variants[0] if task_text_variants else ""
    return {
        "record_id": record_id,
        "source_task_id": str(source.get("task_id", "")).strip(),
        "split": split,
        "task_type": str(source.get("task_type", "")).strip(),
        "task_text": primary_text,
        "task_text_variants": task_text_variants,
        "annotations": annotations,
        "pddl_params": source.get("pddl_params", {}) or {},
        "plan_high_pddl": high_plan,
        "source_trajectory": relative_path,
        "source_sha256": f"sha256:{_sha256_bytes(raw)}",
        "task_text_hash": f"sha256:{_sha256_text(primary_text)}",
        "task_embedding_key": f"sha256:{_sha256_text(primary_text)}",
    }


def _skill_records(
    skills_dir: Path,
    graph: dict[str, Any],
    graph_snapshot_id: str,
) -> list[dict[str, Any]]:
    nodes = graph.get("nodes", {}) or {}
    records = []
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        raw = skill_file.read_bytes()
        text = raw.decode("utf-8")
        metadata, body = _parse_frontmatter(text)
        skill_id = skill_file.parent.name
        node = nodes.get(skill_id, {}) or {}
        description = metadata.get("description") or str(node.get("description", "")).strip()
        name = metadata.get("name") or str(node.get("name", skill_id)).strip()
        embedding_text = "\n".join(part for part in (skill_id, description, body) if part)
        content_hash = _sha256_bytes(raw)
        records.append(
            {
                "skill_id": skill_id,
                "name": name,
                "description": description,
                "skill_body": body,
                "skill_text": text,
                "skill_path": skill_file.relative_to(skills_dir).as_posix(),
                "skill_text_hash": f"sha256:{content_hash}",
                "skill_version": f"content:{content_hash[:16]}",
                "skill_embedding_key": f"sha256:{_sha256_text(embedding_text)}",
                "graph_snapshot_id": graph_snapshot_id,
                "status": node.get("status", ""),
                "tags": node.get("tags", []) or [],
            }
        )
    return records


def extract_static_data(
    alfworld_root: Path | str,
    skills_dir: Path | str,
    graph_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Extract tasks, skills, and a validation report into ``output_root``.

    This function intentionally has no embedding, LLM, or network dependency.
    """
    alfworld_root = Path(alfworld_root).expanduser().resolve()
    skills_dir = Path(skills_dir).expanduser().resolve()
    graph_path = Path(graph_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    for path, kind in ((alfworld_root, "ALFWorld root"), (skills_dir, "skills directory")):
        if not path.is_dir():
            raise FileNotFoundError(f"{kind} does not exist: {path}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"graph file does not exist: {graph_path}")

    graph_raw = graph_path.read_bytes()
    graph = json.loads(graph_raw)
    graph_hash = _sha256_bytes(graph_raw)
    graph_snapshot_id = f"sha256:{graph_hash}"

    task_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    task_ids: Counter[str] = Counter()
    record_ids: Counter[str] = Counter()
    text_hashes: Counter[str] = Counter()
    empty_task_text = 0
    total_annotations = 0

    def iter_tasks() -> Iterable[dict[str, Any]]:
        nonlocal empty_task_text, total_annotations
        for split in SPLITS:
            split_dir = alfworld_root / split
            if not split_dir.is_dir():
                continue
            for path in sorted(split_dir.rglob("traj_data.json")):
                record = _task_record(path, alfworld_root, split)
                task_counts[split] += 1
                task_type_counts[record["task_type"]] += 1
                task_ids[record["source_task_id"]] += 1
                record_ids[record["record_id"]] += 1
                text_hashes[record["task_text_hash"]] += 1
                total_annotations += len(record["annotations"])
                if not record["task_text"]:
                    empty_task_text += 1
                yield record

    tasks_path = output_root / "raw" / "tasks.jsonl"
    task_count = _atomic_jsonl(tasks_path, iter_tasks())

    skills = _skill_records(skills_dir, graph, graph_snapshot_id)
    skills_path = output_root / "raw" / "skills.jsonl"
    skill_count = _atomic_jsonl(skills_path, skills)
    skill_ids = Counter(item["skill_id"] for item in skills)
    skill_hashes = Counter(item["skill_text_hash"] for item in skills)
    graph_nodes = set((graph.get("nodes", {}) or {}).keys())
    file_nodes = set(skill_ids)
    edge_types = Counter(str(edge.get("type", "")) for edge in graph.get("edges", []) or [])

    report = {
        "schema_version": "skilldag_ncf.static_data_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_made": 0,
        "sources": {
            "alfworld_root": str(alfworld_root),
            "skills_dir": str(skills_dir),
            "graph_path": str(graph_path),
        },
        "outputs": {
            "tasks": str(tasks_path),
            "skills": str(skills_path),
        },
        "tasks": {
            "count": task_count,
            "counts_by_split": dict(sorted(task_counts.items())),
            "counts_by_task_type": dict(sorted(task_type_counts.items())),
            "annotation_count": total_annotations,
            "empty_task_text_count": empty_task_text,
            "duplicate_record_ids": sorted(key for key, count in record_ids.items() if count > 1),
            "duplicate_source_task_ids": sorted(key for key, count in task_ids.items() if key and count > 1),
            "duplicate_primary_text_groups": sum(1 for count in text_hashes.values() if count > 1),
        },
        "skills": {
            "count": skill_count,
            "duplicate_skill_ids": sorted(key for key, count in skill_ids.items() if count > 1),
            "duplicate_content_groups": sum(1 for count in skill_hashes.values() if count > 1),
            "missing_skill_files_for_graph_nodes": sorted(graph_nodes - file_nodes),
            "skills_missing_from_graph": sorted(file_nodes - graph_nodes),
            "empty_description_skill_ids": sorted(
                item["skill_id"] for item in skills if not item["description"]
            ),
        },
        "graph": {
            "graph_snapshot_id": graph_snapshot_id,
            "schema_version": graph.get("schema_version"),
            "updated_at": graph.get("updated_at"),
            "node_count": len(graph_nodes),
            "edge_count": len(graph.get("edges", []) or []),
            "edge_counts_by_type": dict(sorted(edge_types.items())),
            "history_count": len(graph.get("history", []) or []),
        },
    }
    report_path = output_root / "reports" / "static_data_report.json"
    report["outputs"]["report"] = str(report_path)
    _atomic_json(report_path, report)
    return report
