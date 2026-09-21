#!/usr/bin/env python3
"""Build the leakage-safe source manifest for SkillDAG + NCF V3.

This script is audit-only. It never calls an API, creates labels, computes
embeddings, or reads online evaluation trajectories. The existing 700-task
official-train split is retained where possible. Any task whose exact text
hash also appears in an official evaluation split is replaced deterministically
by another official-train task of the same task type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "alfworld_task_skill"
RAW_TASKS = DATA_ROOT / "shared/raw/tasks_all_official_splits.jsonl"
RAW_SKILLS = DATA_ROOT / "shared/raw/skills_37.jsonl"
ORIGINAL_SPLIT = DATA_ROOT / "shared/splits/task_level_700.json"
ORIGINAL_SELECTION = DATA_ROOT / "shared/splits/judge_stratified_700.json"
SKILL_EMBEDDINGS = DATA_ROOT / "shared/embeddings/skill_embeddings_37.json"
SKILL_GRAPH = DATA_ROOT / "shared/skillgraph_alfworld.json"
DEFAULT_SKILLS_DIR = (
    ROOT / ".runtime/skilldag/data/skilldag/alfworld_skills"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "task_skill_v3_phase/manifests"

INTERNAL_SPLITS = ("train", "dev", "test")
FORBIDDEN_OFFICIAL_SPLITS = ("valid_train", "valid_seen", "valid_unseen")
SELECTION_SEED = "skilldag-ncf-v3-leakage-safe-replacements-v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def require_unique(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    return result


def stable_candidate_key(record_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}\0{record_id}".encode()).hexdigest()


def count_by_type(
    split_ids: dict[str, list[str]], task_by_id: dict[str, dict[str, Any]]
) -> dict[str, dict[str, int]]:
    task_types = sorted({row["task_type"] for row in task_by_id.values()})
    result: dict[str, dict[str, int]] = {}
    for task_type in task_types:
        result[task_type] = {
            split: sum(
                task_by_id[record_id]["task_type"] == task_type
                for record_id in split_ids[split]
            )
            for split in INTERNAL_SPLITS
        }
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        RAW_TASKS,
        RAW_SKILLS,
        ORIGINAL_SPLIT,
        ORIGINAL_SELECTION,
        SKILL_EMBEDDINGS,
        SKILL_GRAPH,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.skills_dir.is_dir():
        raise FileNotFoundError(args.skills_dir)

    tasks = load_jsonl(RAW_TASKS)
    task_by_id = require_unique(tasks, "record_id")
    split_payload = load_json(ORIGINAL_SPLIT)
    original_ids = {
        split: list(split_payload["record_ids"][split])
        for split in INTERNAL_SPLITS
    }
    original_all = set().union(*(set(ids) for ids in original_ids.values()))

    missing = original_all - set(task_by_id)
    if missing:
        raise ValueError(f"original split references {len(missing)} missing tasks")
    if any(task_by_id[record_id]["split"] != "train" for record_id in original_all):
        raise ValueError("the original 700-task split is not official-train only")

    forbidden_rows = {
        split: [row for row in tasks if row["split"] == split]
        for split in FORBIDDEN_OFFICIAL_SPLITS
    }
    forbidden_text_hashes = {
        str(row["task_text_hash"])
        for rows in forbidden_rows.values()
        for row in rows
    }

    safe_ids: dict[str, list[str]] = {split: [] for split in INTERNAL_SPLITS}
    removed: list[dict[str, str]] = []
    deficits: Counter[tuple[str, str]] = Counter()
    for split in INTERNAL_SPLITS:
        for record_id in original_ids[split]:
            row = task_by_id[record_id]
            if row["task_text_hash"] in forbidden_text_hashes:
                deficits[(split, row["task_type"])] += 1
                removed.append(
                    {
                        "internal_split": split,
                        "record_id": record_id,
                        "task_text_hash": row["task_text_hash"],
                        "task_type": row["task_type"],
                    }
                )
            else:
                safe_ids[split].append(record_id)

    used_hashes = {
        task_by_id[record_id]["task_text_hash"]
        for ids in safe_ids.values()
        for record_id in ids
    }
    candidates_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tasks:
        if row["split"] != "train":
            continue
        if row["record_id"] in original_all:
            continue
        if row["task_text_hash"] in forbidden_text_hashes:
            continue
        candidates_by_type[row["task_type"]].append(row)
    for rows in candidates_by_type.values():
        rows.sort(key=lambda row: stable_candidate_key(row["record_id"]))

    replacements: list[dict[str, str]] = []
    for (split, task_type), required_count in sorted(deficits.items()):
        selected: list[dict[str, Any]] = []
        for row in candidates_by_type[task_type]:
            if row["task_text_hash"] in used_hashes:
                continue
            selected.append(row)
            used_hashes.add(row["task_text_hash"])
            if len(selected) == required_count:
                break
        if len(selected) != required_count:
            raise ValueError(
                f"not enough safe replacements for {split}/{task_type}: "
                f"needed={required_count}, found={len(selected)}"
            )
        for row in selected:
            safe_ids[split].append(row["record_id"])
            replacements.append(
                {
                    "internal_split": split,
                    "record_id": row["record_id"],
                    "task_text_hash": row["task_text_hash"],
                    "task_type": task_type,
                }
            )

    for split in INTERNAL_SPLITS:
        safe_ids[split].sort()

    expected_split_counts = {split: len(original_ids[split]) for split in INTERNAL_SPLITS}
    actual_split_counts = {split: len(safe_ids[split]) for split in INTERNAL_SPLITS}
    if actual_split_counts != expected_split_counts:
        raise ValueError(
            f"safe split counts changed: expected={expected_split_counts}, "
            f"actual={actual_split_counts}"
        )
    original_type_counts = count_by_type(original_ids, task_by_id)
    safe_type_counts = count_by_type(safe_ids, task_by_id)
    if safe_type_counts != original_type_counts:
        raise ValueError("safe replacement changed task-type balance")

    skills = load_jsonl(RAW_SKILLS)
    raw_skill_ids = {row["skill_id"] for row in skills}
    embedding_ids = set(load_json(SKILL_EMBEDDINGS))
    graph = load_json(SKILL_GRAPH)
    graph_ids = set(graph["nodes"])
    document_ids = {
        path.parent.name for path in args.skills_dir.glob("*/SKILL.md")
    }
    if not (
        len(raw_skill_ids) == 37
        and raw_skill_ids == embedding_ids == graph_ids == document_ids
    ):
        raise ValueError(
            "37-skill sources disagree: "
            f"raw={len(raw_skill_ids)}, embeddings={len(embedding_ids)}, "
            f"graph={len(graph_ids)}, documents={len(document_ids)}"
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    input_paths = {
        "raw_tasks": RAW_TASKS,
        "raw_skills": RAW_SKILLS,
        "original_split": ORIGINAL_SPLIT,
        "original_selection": ORIGINAL_SELECTION,
        "skill_embeddings": SKILL_EMBEDDINGS,
        "skill_graph": SKILL_GRAPH,
    }
    source_manifest = {
        "schema_version": "skilldag_ncf.v3.source_manifest.v1",
        "generated_at": generated_at,
        "selection_policy": {
            "allowed_official_split": "train",
            "forbidden_official_splits": list(FORBIDDEN_OFFICIAL_SPLITS),
            "replacement_seed": SELECTION_SEED,
            "replacement_rule": (
                "Replace exact-text-hash collisions with deterministic official-train "
                "tasks of the same task type and internal split."
            ),
        },
        "counts": {
            "original_tasks": len(original_all),
            "selected_tasks": sum(actual_split_counts.values()),
            "removed_exact_text_collisions": len(removed),
            "replacement_tasks": len(replacements),
            "skills": len(raw_skill_ids),
        },
        "record_ids": safe_ids,
        "removed_train_records": removed,
        "replacement_train_records": replacements,
        "source_files": {
            name: {"path": relative(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "skills_dir": relative(args.skills_dir),
    }
    forbidden_manifest = {
        "schema_version": "skilldag_ncf.v3.forbidden_hashes.v1",
        "generated_at": generated_at,
        "purpose": "audit-only denylist; never a model input",
        "splits": {
            split: {
                "record_count": len(rows),
                "record_id_hashes": sorted(
                    sha256_text(str(row["record_id"])) for row in rows
                ),
                "task_text_hashes": sorted(
                    {str(row["task_text_hash"]) for row in rows}
                ),
            }
            for split, rows in forbidden_rows.items()
        },
    }
    split_report = {
        "schema_version": "skilldag_ncf.v3.split_report.v1",
        "generated_at": generated_at,
        "split_counts": actual_split_counts,
        "counts_by_task_type": safe_type_counts,
        "removed_by_internal_split": dict(
            sorted(Counter(row["internal_split"] for row in removed).items())
        ),
        "removed_by_task_type": dict(
            sorted(Counter(row["task_type"] for row in removed).items())
        ),
        "replacement_count": len(replacements),
        "record_id_overlap_across_internal_splits": 0,
        "task_text_hash_overlap_across_internal_splits": 0,
        "forbidden_record_id_overlap": 0,
        "forbidden_task_text_hash_overlap": 0,
        "skill_sources": {
            "raw": len(raw_skill_ids),
            "embeddings": len(embedding_ids),
            "graph_nodes": len(graph_ids),
            "skill_documents": len(document_ids),
            "ids_identical": True,
        },
        "api_calls_made": 0,
    }

    output_dir = args.output_dir
    dump_json(output_dir / "source_manifest.json", source_manifest)
    dump_json(output_dir / "forbidden_hashes.json", forbidden_manifest)
    dump_json(output_dir / "split_report.json", split_report)
    return split_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build(args)
    print("V3 leakage-safe source manifest: BUILT")
    print(f"output: {args.output_dir}")
    print(f"internal splits: {report['split_counts']}")
    print(f"replaced exact-text collisions: {report['replacement_count']}")
    print(f"skills: {report['skill_sources']}")
    print("API calls made: 0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"V3 leakage-safe source manifest: FAILED: {exc}")
        raise SystemExit(1)
