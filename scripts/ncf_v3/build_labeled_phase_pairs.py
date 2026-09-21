#!/usr/bin/env python3
"""Project generic Judge labels onto retrieved phase candidates with calibration."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/alfworld_task_skill"
DEFAULT_PHASES = DATA_ROOT / "task_skill_v3_phase/phases/phases.jsonl"
DEFAULT_CANDIDATES = DATA_ROOT / "task_skill_v3_phase/candidates/phase_candidates.jsonl"
DEFAULT_LABELS = DATA_ROOT / "task_skill_v3_phase/judge/template_labels/labels.jsonl"
DEFAULT_SKILLS = DATA_ROOT / "shared/raw/skills_37.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "task_skill_v3_phase/labeled_pairs"
RETRIEVAL_SOURCES = {"phase_cosine", "full_task_cosine", "v2_ncf_hard"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def primary_capability(description: str) -> str:
    """Use the leading capability statement, not incidental examples later."""
    return re.split(r"(?<=[.!?])\s+", description.strip(), maxsplit=1)[0].lower()


def direct_capability_match(phase: dict[str, Any], description: str) -> bool:
    text = primary_capability(description)
    phase_name = phase["phase_name"]
    if phase_name == "slice_object":
        return bool(re.search(r"\b(slice|slices|slicing|cut|cuts|cutting)\b", text))
    if phase_name in {"locate_object", "locate_distinct_objects"}:
        return bool(re.search(r"\b(find|finds|locate|locates|search|searches)\b", text) and re.search(r"\b(object|objects|item|items|target)\b", text))
    if phase_name == "locate_device":
        return bool(re.search(r"\b(find|finds|locate|locates|search|searches)\b", text) and re.search(r"\b(device|devices|tool|tools|appliance|appliances)\b", text))
    if phase_name == "locate_movable_receptacle":
        return bool(re.search(r"\b(find|finds|locate|locates|search|searches)\b", text) and re.search(r"\b(receptacle|receptacles|container|containers)\b", text))
    if phase_name == "transform_object":
        state = phase["task_structure"]["required_state"]
        forms = {
            "clean": r"\b(clean|cleans|cleaning|cleanliness)\b",
            "cool": r"\b(cool|cools|cooling|temperature)\b",
            "heat": r"\b(heat|heats|heating|temperature)\b",
        }
        return bool(re.search(forms[state], text))
    if phase_name in {"place_object", "place_objects", "place_movable_receptacle", "assemble"}:
        return bool(re.search(r"\b(place|places|placing|put|puts|store|stores|storing|deposit|deposits|move|moves|transport|transports)\b", text) and re.search(r"\b(object|objects|item|items|container|containers|receptacle|receptacles|destination)\b", text))
    if phase_name == "operate_device":
        return bool(re.search(r"\b(operate|operates|toggle|toggles|activate|activates|examine|examines|use|uses)\b", text) and re.search(r"\b(device|appliance|tool|light|object)\b", text))
    if phase_name in {"verify_goal", "verify_count"}:
        return bool(re.search(r"\b(verify|verifies|check|checks|inspect|inspects|confirm|confirms|validate|validates)\b", text))
    raise ValueError(f"unsupported phase for calibration: {phase_name}")


def calibrated_grade(raw_grade: int | str, direct: bool) -> tuple[int | str, str]:
    if raw_grade == "uncertain":
        return raw_grade, "unchanged_uncertain"
    grade = int(raw_grade)
    if direct and grade != 2:
        return 2, f"upgrade_{grade}_to_2_explicit_direct_capability"
    if not direct and grade == 2:
        return 1, "downgrade_2_to_1_no_explicit_direct_capability"
    return grade, "unchanged"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases-path", type=Path, default=DEFAULT_PHASES)
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--skills-path", type=Path, default=DEFAULT_SKILLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    phases = {row["phase_id"]: row for row in load_jsonl(args.phases_path)}
    skills = {row["skill_id"]: row for row in load_jsonl(args.skills_path)}
    template_labels = {
        (row["template_id"], row["skill_id"]): row
        for row in load_jsonl(args.labels_path)
    }
    candidates = load_jsonl(args.candidates_path)
    expected_templates = {f"{row['task_type']}::{row['phase_name']}" for row in phases.values()}
    if {template_id for template_id, _ in template_labels} != expected_templates:
        raise ValueError("template Judge labels do not cover all phase templates")

    output_rows = []
    calibration_counts: Counter[str] = Counter()
    raw_grade_counts: Counter[str] = Counter()
    grade_counts: Counter[str] = Counter()
    phase_candidate_counts: Counter[str] = Counter()
    grade2_by_phase: Counter[str] = Counter()
    excluded_graph_only = 0
    template_calibrated: dict[tuple[str, str], tuple[int | str, str, bool]] = {}

    for candidate in candidates:
        if not (set(candidate["candidate_sources"]) & RETRIEVAL_SOURCES):
            excluded_graph_only += 1
            continue
        phase = phases[candidate["phase_id"]]
        skill = skills[candidate["skill_id"]]
        template_id = f"{phase['task_type']}::{phase['phase_name']}"
        label = template_labels[(template_id, candidate["skill_id"])]
        calibration_key = (template_id, candidate["skill_id"])
        if calibration_key not in template_calibrated:
            direct = direct_capability_match(phase, skill["description"])
            grade, action = calibrated_grade(label["grade"], direct)
            template_calibrated[calibration_key] = (grade, action, direct)
        grade, action, direct = template_calibrated[calibration_key]
        confidence = float(label["confidence"])
        if action not in {"unchanged", "unchanged_uncertain"}:
            confidence = min(confidence, 0.85)
        phase_candidate_counts[phase["phase_id"]] += 1
        grade2_by_phase[phase["phase_id"]] += int(grade == 2)
        calibration_counts[action] += 1
        raw_grade_counts[str(label["grade"])] += 1
        grade_counts[str(grade)] += 1
        output_rows.append(
            {
                "schema_version": "skilldag_ncf.v3.labeled_phase_skill_pair.v1",
                "pair_id": f"{phase['phase_id']}::{candidate['skill_id']}",
                "task_record_id": phase["task_record_id"],
                "phase_id": phase["phase_id"],
                "template_id": template_id,
                "internal_split": phase["internal_split"],
                "task_type": phase["task_type"],
                "phase_name": phase["phase_name"],
                "phase_group": phase["phase_group"],
                "skill_id": candidate["skill_id"],
                "candidate_sources": candidate["candidate_sources"],
                "phase_cosine_rank": candidate.get("phase_cosine_rank"),
                "phase_cosine_score": candidate.get("phase_cosine_score"),
                "full_task_cosine_rank": candidate.get("full_task_cosine_rank"),
                "full_task_cosine_score": candidate.get("full_task_cosine_score"),
                "v2_ncf_hard_rank": candidate.get("v2_ncf_hard_rank"),
                "v2_ncf_hard_score": candidate.get("v2_ncf_hard_score"),
                "graph_paths": candidate.get("graph_paths", []),
                "raw_judge_grade": label["grade"],
                "target_grade": grade,
                "label_confidence": confidence,
                "judge_reason": label["reason"],
                "explicit_direct_capability": direct,
                "calibration_action": action,
                "label_source": "generic_phase_judge_plus_explicit_capability_calibration",
                "uses_expert_plan": False,
                "uses_eval_data": False,
            }
        )

    output_rows.sort(key=lambda row: (row["internal_split"], row["phase_id"], row["skill_id"]))
    phases_without_grade2 = sorted(phase_id for phase_id in phases if grade2_by_phase[phase_id] == 0)
    by_group = defaultdict(Counter)
    for row in output_rows:
        by_group[row["phase_group"]][str(row["target_grade"])] += 1
    report = {
        "schema_version": "skilldag_ncf.v3.labeled_phase_pairs_report.v1",
        "task_count": len({row["task_record_id"] for row in output_rows}),
        "phase_count": len(phase_candidate_counts),
        "pair_count": len(output_rows),
        "excluded_graph_only_pair_count": excluded_graph_only,
        "candidates_per_phase": {
            "min": min(phase_candidate_counts.values()),
            "mean": sum(phase_candidate_counts.values()) / len(phase_candidate_counts),
            "max": max(phase_candidate_counts.values()),
        },
        "raw_judge_grade_counts": dict(sorted(raw_grade_counts.items())),
        "target_grade_counts": dict(sorted(grade_counts.items())),
        "target_grades_by_phase_group": {key: dict(sorted(value.items())) for key, value in sorted(by_group.items())},
        "calibration_action_counts": dict(sorted(calibration_counts.items())),
        "phases_without_grade2_count": len(phases_without_grade2),
        "phases_without_grade2_by_name": dict(sorted(Counter(phases[phase_id]["phase_name"] for phase_id in phases_without_grade2).items())),
        "uses_expert_plan": False,
        "uses_eval_data": False,
        "outputs": {"pairs": "pairs.jsonl", "phases_without_grade2": "phases_without_grade2.json"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pairs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    (args.output_dir / "phases_without_grade2.json").write_text(json.dumps(phases_without_grade2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
