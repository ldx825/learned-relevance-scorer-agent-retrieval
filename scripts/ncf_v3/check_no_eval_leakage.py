#!/usr/bin/env python3
"""Fail-closed audit for the SkillDAG + NCF V3 source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "alfworld_task_skill"
RAW_TASKS = DATA_ROOT / "shared/raw/tasks_all_official_splits.jsonl"
RAW_SKILLS = DATA_ROOT / "shared/raw/skills_37.jsonl"
SKILL_EMBEDDINGS = DATA_ROOT / "shared/embeddings/skill_embeddings_37.json"
SKILL_GRAPH = DATA_ROOT / "shared/skillgraph_alfworld.json"
DEFAULT_MANIFEST_DIR = DATA_ROOT / "task_skill_v3_phase/manifests"
INTERNAL_SPLITS = ("train", "dev", "test")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def resolve_source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def check(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest_dir / "source_manifest.json"
    forbidden_path = args.manifest_dir / "forbidden_hashes.json"
    report_path = args.manifest_dir / "split_report.json"
    for path in (manifest_path, forbidden_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = load_json(manifest_path)
    forbidden = load_json(forbidden_path)
    saved_report = load_json(report_path)
    tasks = load_jsonl(RAW_TASKS)
    task_by_id = {row["record_id"]: row for row in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("raw task archive contains duplicate record_id values")

    split_ids = {
        split: list(manifest["record_ids"][split]) for split in INTERNAL_SPLITS
    }
    split_sets = {split: set(ids) for split, ids in split_ids.items()}
    if any(len(split_sets[split]) != len(split_ids[split]) for split in INTERNAL_SPLITS):
        raise ValueError("a V3 internal split contains duplicate record IDs")

    expected_counts = {"train": 560, "dev": 70, "test": 70}
    actual_counts = {split: len(ids) for split, ids in split_sets.items()}
    if actual_counts != expected_counts:
        raise ValueError(
            f"unexpected V3 split counts: expected={expected_counts}, actual={actual_counts}"
        )

    selected = set().union(*split_sets.values())
    if len(selected) != 700:
        raise ValueError(f"expected 700 unique tasks, found {len(selected)}")
    missing = selected - set(task_by_id)
    if missing:
        raise ValueError(f"manifest references {len(missing)} missing task IDs")
    non_train = [record_id for record_id in selected if task_by_id[record_id]["split"] != "train"]
    if non_train:
        raise ValueError(f"manifest contains {len(non_train)} non-train tasks")

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} record ID overlap: {len(overlap)}")

    text_hash_sets = {
        split: {task_by_id[record_id]["task_text_hash"] for record_id in ids}
        for split, ids in split_sets.items()
    }
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = text_hash_sets[left] & text_hash_sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} task_text_hash overlap: {len(overlap)}")

    forbidden_record_hashes = {
        value
        for payload in forbidden["splits"].values()
        for value in payload["record_id_hashes"]
    }
    forbidden_text_hashes = {
        value
        for payload in forbidden["splits"].values()
        for value in payload["task_text_hashes"]
    }
    record_overlap = {
        record_id for record_id in selected if sha256_text(record_id) in forbidden_record_hashes
    }
    text_overlap = {
        task_by_id[record_id]["task_text_hash"]
        for record_id in selected
        if task_by_id[record_id]["task_text_hash"] in forbidden_text_hashes
    }
    if record_overlap:
        raise ValueError(f"official evaluation record ID overlap: {len(record_overlap)}")
    if text_overlap:
        raise ValueError(f"official evaluation task text overlap: {len(text_overlap)}")

    type_counts: dict[str, dict[str, int]] = {}
    for task_type in sorted({task_by_id[record_id]["task_type"] for record_id in selected}):
        type_counts[task_type] = {
            split: sum(
                task_by_id[record_id]["task_type"] == task_type
                for record_id in split_ids[split]
            )
            for split in INTERNAL_SPLITS
        }
    expected_per_type = {"train": 80, "dev": 10, "test": 10}
    unbalanced = {
        task_type: counts
        for task_type, counts in type_counts.items()
        if counts != expected_per_type
    }
    if len(type_counts) != 7 or unbalanced:
        raise ValueError(
            f"task types are not 7 x (80/10/10): types={len(type_counts)}, "
            f"unbalanced={unbalanced}"
        )

    for source in manifest["source_files"].values():
        path = resolve_source_path(source["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        current_hash = sha256_file(path)
        if current_hash != source["sha256"]:
            raise ValueError(f"source file changed after manifest build: {path}")

    raw_skill_ids = {row["skill_id"] for row in load_jsonl(RAW_SKILLS)}
    embedding_ids = set(load_json(SKILL_EMBEDDINGS))
    graph_ids = set(load_json(SKILL_GRAPH)["nodes"])
    skills_dir = resolve_source_path(manifest["skills_dir"])
    document_ids = {path.parent.name for path in skills_dir.glob("*/SKILL.md")}
    if not (
        len(raw_skill_ids) == 37
        and raw_skill_ids == embedding_ids == graph_ids == document_ids
    ):
        raise ValueError(
            "skill source mismatch: "
            f"raw={len(raw_skill_ids)}, embeddings={len(embedding_ids)}, "
            f"graph={len(graph_ids)}, documents={len(document_ids)}"
        )

    removed = manifest["removed_train_records"]
    replacements = manifest["replacement_train_records"]
    if len(removed) != len(replacements):
        raise ValueError(
            f"removed/replacement mismatch: {len(removed)} != {len(replacements)}"
        )
    replacement_type_counts = Counter(
        (row["internal_split"], row["task_type"]) for row in replacements
    )
    removed_type_counts = Counter(
        (row["internal_split"], row["task_type"]) for row in removed
    )
    if replacement_type_counts != removed_type_counts:
        raise ValueError("replacement tasks do not preserve split/task-type cells")

    result = {
        "split_counts": actual_counts,
        "task_type_count": len(type_counts),
        "record_id_overlap": 0,
        "task_text_hash_overlap": 0,
        "forbidden_record_id_overlap": 0,
        "forbidden_task_text_hash_overlap": 0,
        "replacement_count": len(replacements),
        "skill_count": len(raw_skill_ids),
        "api_calls_made": 0,
    }
    if saved_report["split_counts"] != actual_counts:
        raise ValueError("saved split report disagrees with fresh audit")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    return parser.parse_args()


def main() -> int:
    result = check(parse_args())
    print("[OK] Official source split: train only")
    print(f"[OK] Selected tasks: {sum(result['split_counts'].values())}")
    print(f"[OK] Internal split: {result['split_counts']}")
    print(f"[OK] Record ID overlap: {result['record_id_overlap']}")
    print(f"[OK] Task text hash overlap: {result['task_text_hash_overlap']}")
    print(f"[OK] Official evaluation record overlap: {result['forbidden_record_id_overlap']}")
    print(f"[OK] Official evaluation text overlap: {result['forbidden_task_text_hash_overlap']}")
    print(f"[OK] Replaced exact-text collisions: {result['replacement_count']}")
    print(f"[OK] Seven task types are balanced: {result['task_type_count'] == 7}")
    print(f"[OK] Skills: documents=embeddings=graph=raw={result['skill_count']}")
    print("[PASS] V3 source data is leakage-safe")
    print("API calls made: 0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[FAILED] V3 leakage audit: {exc}")
        raise SystemExit(1)
