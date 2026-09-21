"""Deterministic cleaning and numeric preparation for GoS V2 weak labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .audit_v2 import TRANSFORM_SKILLS, requested_transformations
from .data_v2 import atomic_json, atomic_jsonl, load_jsonl


SPLIT_CODES = {"train": 0, "dev": 1, "test": 2}
ROLE_VALUES = {
    "primary": (2, 1.0, 1.0, True),
    "support": (1, 0.7, 0.7, True),
    "irrelevant": (0, 0.0, 0.7, True),
    "harmful": (0, 0.0, 1.0, True),
    "uncertain": (None, None, 0.0, False),
}
PREFERRED_SKILLS = {
    "clean": ("alfworld-clean-object",),
    "cool": ("alfworld-object-cooler", "alfworld-temperature-regulator"),
    "heat": (
        "alfworld-object-heater",
        "alfworld-heat-object-with-appliance",
        "alfworld-temperature-regulator",
    ),
}


def _apply_role(row: dict[str, Any], role: str, flag: str) -> None:
    grade, score, weight, trainable = ROLE_VALUES[role]
    row["role"] = role
    row["target_grade"] = grade
    row["target_score"] = score
    row["sample_weight"] = weight
    row["trainable"] = trainable
    row.setdefault("quality_flags", [])
    row["quality_flags"] = sorted(set(row["quality_flags"]) | {flag})


def _candidate_scores(path: Path) -> dict[tuple[str, str], float]:
    return {
        (row["task_record_id"], row["skill_id"]): float(
            row.get("semantic_cosine") or -1.0
        )
        for row in load_jsonl(path)
    }


def clean_gos_v2_labels(
    tasks_path: Path | str,
    labels_path: Path | str,
    candidates_path: Path | str,
    output_path: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    """Repair only ontology-checkable transformation contradictions.

    The original Judge output remains immutable.  Every changed row stores its
    original role, and the report distinguishes promotions, demotions, and
    candidate-recall repairs.
    """
    tasks = {row["record_id"]: row for row in load_jsonl(Path(tasks_path))}
    rows = [dict(row) for row in load_jsonl(Path(labels_path))]
    scores = _candidate_scores(Path(candidates_path))
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["task_record_id"]][row["skill_id"]] = row

    changes: list[dict[str, Any]] = []
    for task_id, task_rows in grouped.items():
        requested = requested_transformations(tasks[task_id])
        requested_skill_ids = (
            set().union(*(TRANSFORM_SKILLS[value] for value in requested))
            if requested
            else set()
        )

        # A positive operation-specific skill is contradictory when neither
        # the task text nor the ALFWorld goal type asks for that operation.
        for operation, skill_ids in TRANSFORM_SKILLS.items():
            if operation in requested:
                continue
            for skill_id in skill_ids:
                if skill_id in requested_skill_ids:
                    continue
                row = task_rows.get(skill_id)
                if row and row["role"] in {"primary", "support"}:
                    old_role = row["role"]
                    row["audit_original_role"] = old_role
                    _apply_role(row, "irrelevant", "semantic_rule_demoted")
                    changes.append(
                        {
                            "task_record_id": task_id,
                            "operation": operation,
                            "skill_id": skill_id,
                            "from": old_role,
                            "to": "irrelevant",
                            "reason": "operation_not_requested",
                        }
                    )

        for operation in sorted(requested):
            family = TRANSFORM_SKILLS[operation]
            if any(
                task_rows.get(skill_id, {}).get("role") == "primary"
                for skill_id in family
            ):
                continue
            available = [skill_id for skill_id in family if skill_id in task_rows]
            if available:
                preferred_order = {
                    skill_id: index
                    for index, skill_id in enumerate(PREFERRED_SKILLS[operation])
                }
                chosen = max(
                    available,
                    key=lambda skill_id: (
                        scores.get((task_id, skill_id), -1.0),
                        -preferred_order.get(skill_id, 999),
                        skill_id,
                    ),
                )
                row = task_rows[chosen]
                old_role = row["role"]
                row["audit_original_role"] = old_role
                _apply_role(row, "primary", "semantic_rule_promoted")
                changes.append(
                    {
                        "task_record_id": task_id,
                        "operation": operation,
                        "skill_id": chosen,
                        "from": old_role,
                        "to": "primary",
                        "reason": "required_operation_missing_from_primary",
                    }
                )
                continue

            # Candidate generation missed the whole capability family.  Add
            # the most specific ontology skill instead of silently dropping
            # the task from training/evaluation.
            chosen = PREFERRED_SKILLS[operation][0]
            row = {
                "dataset_version": "task_skill_v2_gos_clean",
                "task_record_id": task_id,
                "skill_id": chosen,
                "role": "primary",
                "target_grade": 2,
                "target_score": 1.0,
                "sample_weight": 1.0,
                "trainable": True,
                "task_evidence": tasks[task_id]["task_text"],
                "task_evidence_valid": True,
                "source_model": "deterministic_semantic_audit",
                "source_prompt_version": "none",
                "expert_plan_visible": False,
                "quality_flags": ["candidate_recall_repair", "semantic_rule_promoted"],
                "audit_original_role": "missing",
            }
            rows.append(row)
            task_rows[chosen] = row
            changes.append(
                {
                    "task_record_id": task_id,
                    "operation": operation,
                    "skill_id": chosen,
                    "from": "missing",
                    "to": "primary",
                    "reason": "required_capability_absent_from_candidate_pool",
                }
            )

    rows.sort(key=lambda row: (row["task_record_id"], row["skill_id"]))
    weight_normalization_count = 0
    for row in rows:
        expected_weight = ROLE_VALUES[row["role"]][2]
        old_weight = float(row.get("sample_weight") or 0.0)
        if abs(old_weight - expected_weight) > 1e-12:
            row["audit_original_sample_weight"] = old_weight
            row["sample_weight"] = expected_weight
            row.setdefault("quality_flags", [])
            row["quality_flags"] = sorted(
                set(row["quality_flags"]) | {"training_weight_normalized"}
            )
            weight_normalization_count += 1
    atomic_jsonl(Path(output_path), rows)
    report = {
        "schema_version": "gos_ncf.label_cleaning.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_made": 0,
        "task_count": len(grouped),
        "pair_count_before": sum(len(values) for values in grouped.values())
        - sum(change["from"] == "missing" for change in changes),
        "pair_count_after": len(rows),
        "change_count": len(changes),
        "weight_normalization_count": weight_normalization_count,
        "sample_weight_policy": {
            "primary": 1.0,
            "support": 0.7,
            "irrelevant": 0.7,
            "harmful": 1.0,
            "uncertain": 0.0,
            "note": "Weights express confidence/importance, not target relevance; negative examples must not have zero weight.",
        },
        "change_types": dict(
            Counter(
                "candidate_added"
                if change["from"] == "missing"
                else "promoted"
                if change["to"] == "primary"
                else "demoted"
                for change in changes
            )
        ),
        "changes": changes,
        "outputs": {"labels": str(Path(output_path)), "report": str(Path(report_path))},
    }
    atomic_json(Path(report_path), report)
    return report


def _load_task_embeddings(cache_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(cache_dir.glob("shard_*.json")):
        result.update(json.loads(path.read_text(encoding="utf-8")))
    return result


def prepare_gos_v2_arrays(
    tasks_path: Path | str,
    skills_path: Path | str,
    labels_path: Path | str,
    splits_path: Path | str,
    task_embedding_cache_dir: Path | str,
    skill_embeddings_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Create leakage-free arrays using existing local embeddings only."""
    tasks_all = {row["record_id"]: row for row in load_jsonl(Path(tasks_path))}
    skills_all = {row["skill_id"]: row for row in load_jsonl(Path(skills_path))}
    labels = load_jsonl(Path(labels_path))
    selected_task_ids = sorted({row["task_record_id"] for row in labels})
    used_skill_ids = sorted(skills_all)
    task_cache = _load_task_embeddings(Path(task_embedding_cache_dir))
    skill_cache = json.loads(Path(skill_embeddings_path).read_text(encoding="utf-8"))
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))["record_ids"]
    split_by_task = {
        task_id: split for split, ids in splits.items() for task_id in ids
    }
    missing_splits = set(selected_task_ids) - set(split_by_task)
    if missing_splits:
        raise ValueError(f"missing split assignments: {len(missing_splits)}")

    task_embeddings = np.asarray(
        [task_cache[task_id]["embedding"] for task_id in selected_task_ids],
        dtype=np.float32,
    )
    skill_embeddings = np.asarray(
        [skill_cache[skill_id]["embedding"] for skill_id in used_skill_ids],
        dtype=np.float32,
    )
    task_norm = task_embeddings / np.maximum(
        np.linalg.norm(task_embeddings, axis=1, keepdims=True), 1e-12
    )
    skill_norm = skill_embeddings / np.maximum(
        np.linalg.norm(skill_embeddings, axis=1, keepdims=True), 1e-12
    )
    cosine = task_norm @ skill_norm.T
    task_index = {value: index for index, value in enumerate(selected_task_ids)}
    skill_index = {value: index for index, value in enumerate(used_skill_ids)}

    pair_task_indices = np.asarray(
        [task_index[row["task_record_id"]] for row in labels], dtype=np.int32
    )
    pair_skill_indices = np.asarray(
        [skill_index[row["skill_id"]] for row in labels], dtype=np.int16
    )
    target_grade = np.asarray(
        [-1 if row.get("target_grade") is None else int(row["target_grade"]) for row in labels],
        dtype=np.int8,
    )
    target_relevance = np.asarray(
        [0.0 if row.get("target_score") is None else float(row["target_score"]) for row in labels],
        dtype=np.float32,
    )
    sample_weight = np.asarray(
        [float(row.get("sample_weight") or 0.0) for row in labels], dtype=np.float32
    )
    split_codes = np.asarray(
        [SPLIT_CODES[split_by_task[row["task_record_id"]]] for row in labels],
        dtype=np.int8,
    )
    pair_features = cosine[pair_task_indices, pair_skill_indices, None].astype(np.float32)
    pair_ids = np.asarray(
        [
            "pair:"
            + hashlib.sha256(
                f"task_skill_v2_gos_clean:{row['task_record_id']}:{row['skill_id']}".encode()
            ).hexdigest()[:24]
            for row in labels
        ]
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        task_ids=np.asarray(selected_task_ids),
        skill_ids=np.asarray(used_skill_ids),
        pair_ids=pair_ids,
        task_embeddings=task_embeddings,
        skill_embeddings=skill_embeddings,
        pair_task_indices=pair_task_indices,
        pair_skill_indices=pair_skill_indices,
        pair_features=pair_features,
        target_grade=target_grade,
        target_relevance=target_relevance,
        sample_weight=sample_weight,
        split_codes=split_codes,
    )
    report = {
        "schema_version": "gos_ncf.model_data.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_calls_made": 0,
        "task_count": len(selected_task_ids),
        "skill_count": len(used_skill_ids),
        "pair_count": len(labels),
        "trainable_pair_count": int(np.sum(target_grade >= 0)),
        "embedding_dimension": int(task_embeddings.shape[1]),
        "split_task_counts": dict(
            Counter(split_by_task[task_id] for task_id in selected_task_ids)
        ),
        "split_pair_counts": dict(
            Counter(split_by_task[row["task_record_id"]] for row in labels)
        ),
        "grade_counts": dict(Counter(map(int, target_grade))),
        "outputs": {
            "arrays": str(arrays_path),
            "metadata": str(output_dir / "metadata.json"),
        },
    }
    atomic_json(output_dir / "metadata.json", report)
    return report
