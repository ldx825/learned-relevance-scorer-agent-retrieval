#!/usr/bin/env python3
"""Fail-closed audit for V3 labeled phase-skill pairs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/alfworld_task_skill/task_skill_v3_phase"
PHASES = DATA / "phases/phases.jsonl"
CANDIDATES = DATA / "candidates/phase_candidates.jsonl"
LABELS = DATA / "judge/template_labels/labels.jsonl"
PAIRS = DATA / "labeled_pairs/pairs.jsonl"
REPORT = DATA / "labeled_pairs/report.json"
RETRIEVAL_SOURCES = {"phase_cosine", "full_task_cosine", "v2_ncf_hard"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    phases = {row["phase_id"]: row for row in load_jsonl(PHASES)}
    candidates = load_jsonl(CANDIDATES)
    labels = load_jsonl(LABELS)
    pairs = load_jsonl(PAIRS)
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    expected = {
        (row["phase_id"], row["skill_id"])
        for row in candidates
        if set(row["candidate_sources"]) & RETRIEVAL_SOURCES
    }
    actual = [(row["phase_id"], row["skill_id"]) for row in pairs]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate phase-skill pairs")
    if set(actual) != expected:
        raise ValueError("labeled pairs differ from the three retrieval branches")
    if len(labels) != 33 * 37 or len({(row["template_id"], row["skill_id"]) for row in labels}) != 33 * 37:
        raise ValueError("template Judge table must contain exactly 33 x 37 judgments")
    if set(phases) != {row["phase_id"] for row in pairs}:
        raise ValueError("some deterministic phases have no labeled candidates")

    by_phase = defaultdict(list)
    for row in pairs:
        by_phase[row["phase_id"]].append(row)
        if row["phase_id"] not in phases:
            raise ValueError(f"unknown phase ID: {row['phase_id']}")
        if row["target_grade"] not in {0, 1, 2}:
            raise ValueError(f"invalid target grade: {row['pair_id']}")
        if row["uses_expert_plan"] or row["uses_eval_data"]:
            raise ValueError(f"forbidden provenance flag: {row['pair_id']}")
        if not (set(row["candidate_sources"]) & RETRIEVAL_SOURCES):
            raise ValueError(f"graph-only candidate survived: {row['pair_id']}")
        if not 0 <= row["label_confidence"] <= 1:
            raise ValueError(f"invalid label confidence: {row['pair_id']}")

    for phase_id, rows in by_phase.items():
        cosine_ranks = sorted(
            row["phase_cosine_rank"]
            for row in rows
            if row["phase_cosine_rank"] is not None
        )
        if cosine_ranks != list(range(1, 13)):
            raise ValueError(f"phase cosine Top-12 is incomplete: {phase_id}")

    task_splits = defaultdict(set)
    for row in pairs:
        task_splits[row["internal_split"]].add(row["task_record_id"])
    split_counts = {key: len(value) for key, value in sorted(task_splits.items())}
    if split_counts != {"dev": 70, "test": 70, "train": 560}:
        raise ValueError(f"unexpected task split counts: {split_counts}")
    grade_counts = Counter(str(row["target_grade"]) for row in pairs)
    no_grade2 = [
        phase_id
        for phase_id, rows in by_phase.items()
        if not any(row["target_grade"] == 2 for row in rows)
    ]
    no_grade2_names = Counter(phases[phase_id]["phase_name"] for phase_id in no_grade2)
    if any(
        row["target_grade"] == 2
        for phase_id, rows in by_phase.items()
        if phases[phase_id]["phase_name"] == "slice_object"
        for row in rows
    ):
        raise ValueError("slice phase has a fabricated direct skill despite the library gap")
    if report["pair_count"] != len(pairs) or report["target_grade_counts"] != dict(sorted(grade_counts.items())):
        raise ValueError("labeled-pair report disagrees with data")

    print(f"[OK] Template judgments: {len(labels)} = 33 x 37")
    print(f"[OK] Labeled pairs: {len(pairs)} unique")
    print(f"[OK] Task splits: {split_counts}")
    print("[OK] Every phase retains exact phase-cosine ranks 1..12")
    print("[OK] Graph-only candidates excluded; expert/eval data absent")
    print(f"[OK] Target grades: {dict(sorted(grade_counts.items()))}")
    print(f"[KNOWN GAP] Phases without grade 2: {len(no_grade2)} {dict(sorted(no_grade2_names.items()))}")
    print("[PASS] V3 labeled phase-skill pairs are structurally valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
