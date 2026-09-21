#!/usr/bin/env python3
"""Validate the private ALFWorld task-skill archive without network/API calls."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "alfworld_task_skill"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"empty required file: {path}")


def main() -> int:
    required = [
        DATA / "shared/raw/tasks_all_official_splits.jsonl",
        DATA / "shared/raw/skills_37.jsonl",
        DATA / "shared/splits/judge_stratified_700.json",
        DATA / "shared/splits/task_level_700.json",
        DATA / "shared/embeddings/skill_embeddings_37.json",
        DATA / "shared/skillgraph_alfworld.json",
        DATA / "task_skill_v2_gos_clean/candidates/task_skill_v2_gos_candidates.jsonl",
        DATA / "task_skill_v2_gos_clean/labels/judge_raw.jsonl",
        DATA / "task_skill_v2_gos_clean/labels/judge_clean.jsonl",
        DATA / "task_skill_v2_gos_clean/model_data/arrays.npz",
        DATA / "task_skill_v2_gos_clean/model_data/metadata.json",
    ]
    for path in required:
        require(path)

    tasks = load_jsonl(required[0])
    skills = load_jsonl(required[1])
    selection_payload = json.loads(required[2].read_text(encoding="utf-8"))
    split_payload = json.loads(required[3].read_text(encoding="utf-8"))
    split_ids = split_payload["record_ids"]
    split_sets = {name: set(ids) for name, ids in split_ids.items()}

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} task overlap: {len(overlap)}")

    task_by_id = {row["record_id"]: row for row in tasks}
    selected = set().union(*split_sets.values())
    missing = selected - set(task_by_id)
    if missing:
        raise ValueError(f"split references missing tasks: {len(missing)}")
    if any(task_by_id[task_id]["split"] != "train" for task_id in selected):
        raise ValueError("the 700-task recommendation dataset must come from official train")

    text_hash_splits: dict[str, set[str]] = {}
    for split, ids in split_sets.items():
        for task_id in ids:
            text_hash = task_by_id[task_id]["task_text_hash"]
            text_hash_splits.setdefault(text_hash, set()).add(split)
    leaked_hashes = [key for key, values in text_hash_splits.items() if len(values) > 1]
    if leaked_hashes:
        raise ValueError(f"task_text_hash leakage across splits: {len(leaked_hashes)}")

    clean_labels = load_jsonl(
        DATA / "task_skill_v2_gos_clean/labels/judge_clean.jsonl"
    )
    label_task_ids = {row["task_record_id"] for row in clean_labels}
    if label_task_ids != selected:
        raise ValueError(
            f"label/split task mismatch: labels={len(label_task_ids)} splits={len(selected)}"
        )

    judge_responses = list(
        (DATA / "task_skill_v2_gos_clean/judge_cache/responses").glob("*.json")
    )
    if set(selection_payload["record_ids"]) != selected:
        raise ValueError("stratified Judge selection differs from the 700 split tasks")

    task_embedding_shards = list(
        (
            DATA
            / "shared/embeddings/task_embeddings_official_train"
        ).glob("shard_*.json")
    )
    metadata = json.loads(
        (
            DATA / "task_skill_v2_gos_clean/model_data/metadata.json"
        ).read_text(encoding="utf-8")
    )

    official_counts = Counter(row["split"] for row in tasks)
    role_counts = Counter(row["role"] for row in clean_labels)
    print("ALFWorld task-skill archive: OK")
    print(f"archive: {DATA}")
    print(f"official tasks: {dict(sorted(official_counts.items()))}")
    print(f"shared skills: {len(skills)}")
    print(f"internal task splits: { {key: len(value) for key, value in split_sets.items()} }")
    print(f"clean V2 pairs: {len(clean_labels)}; roles: {dict(role_counts)}")
    print(f"model metadata pairs: {metadata['pair_count']}")
    print(f"Judge cached tasks: {len(judge_responses)}")
    print(f"task embedding shards: {len(task_embedding_shards)}")
    print("API calls made: 0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ALFWorld task-skill archive: FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
