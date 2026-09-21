#!/usr/bin/env python3
"""Build train-only typed hard pairs for TravelPlanner atomic retrieval.

The online atomic retriever compares operations only within the required
operation type.  The first atomic dataset mostly compared an exact operation
against cross-type complements, so its training distribution did not match
that online decision.  This builder keeps a compact, uniform four-candidate
contract per subgoal whenever the operation bank permits it:

* one structurally closest, compatible operation (Grade 2);
* two different compatible operations of the same type (Grade 1);
* one high-cosine conflicting operation (Grade 0), preferring the same type.

All inputs are derived from the public TravelPlanner train split.  Official
validation/test tasks, trajectories, rewards, and gold plans are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from travelplanner_atomic_memory import seed_grade


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def structural_distance(task: dict[str, Any], memory: dict[str, Any]) -> float:
    return (
        abs(int(task["days"]) - int(memory["days"])) / 7.0
        + abs(int(task["visiting_city_number"]) - int(memory["visiting_city_number"])) / 3.0
        + abs(int(task["people_number"]) - int(memory["people_number"])) / 10.0
        + abs(float(task["budget_per_person_day"]) - float(memory["budget_per_person_day"]))
        / max(float(task["budget_per_person_day"]), 1.0)
    )


def procedure_fingerprint(operation: dict[str, Any]) -> str:
    """Collapse exact formatting variants without conflating real paraphrases."""
    procedure = str(operation.get("procedure") or operation.get("operation_text") or "")
    return " ".join(re.findall(r"[a-z0-9]+", procedure.lower()))


def pair_row(
    subgoal: dict[str, Any],
    operation: dict[str, Any],
    *,
    grade: int,
    role: str,
    reason: str,
    cosine: float,
    distance: float,
) -> dict[str, Any]:
    return {
        "family_id": subgoal["family_id"],
        "final_grade": grade,
        "judge_required": False,
        "label_source": "deterministic_typed_hard_contract_v1",
        "model_split": subgoal["model_split"],
        "operation_id": operation["operation_id"],
        "operation_key": operation["operation_key"],
        "operation_text": operation["operation_text"],
        "parent_memory_id": int(operation["parent_memory_id"]),
        "selection_role": role,
        "seed_reason": reason,
        "subgoal_id": subgoal["subgoal_id"],
        "subgoal_operation_key": subgoal["operation_key"],
        "subgoal_query": subgoal["subgoal_query"],
        "typed_candidate": operation["operation_key"] == subgoal["operation_key"],
        "baseline_cosine": cosine,
        "structural_distance": distance,
        "validation_or_test_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgoals", type=Path, required=True)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--embedding-arrays", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--same-type-supports", type=int, default=2)
    args = parser.parse_args()

    subgoals = read_jsonl(args.subgoals)
    operations = read_jsonl(args.operations)
    if not subgoals or not operations:
        raise ValueError("empty subgoal or operation input")
    if any(row.get("validation_or_test_used") is not False for row in subgoals + operations):
        raise ValueError("validation/test-derived input detected")

    arrays = np.load(args.embedding_arrays, allow_pickle=False)
    subgoal_ids = [str(value) for value in arrays["subgoal_ids"]]
    operation_ids = [str(value) for value in arrays["operation_ids"]]
    subgoal_index = {value: index for index, value in enumerate(subgoal_ids)}
    operation_index = {value: index for index, value in enumerate(operation_ids)}
    if {row["subgoal_id"] for row in subgoals} != set(subgoal_index):
        raise ValueError("subgoal embedding IDs do not match input")
    if {row["operation_id"] for row in operations} != set(operation_index):
        raise ValueError("operation embedding IDs do not match input")

    task_vectors = arrays["task_embeddings"].astype(np.float32)
    operation_vectors = arrays["memory_embeddings"].astype(np.float32)
    task_vectors /= np.maximum(np.linalg.norm(task_vectors, axis=1, keepdims=True), 1e-12)
    operation_vectors /= np.maximum(np.linalg.norm(operation_vectors, axis=1, keepdims=True), 1e-12)

    operations_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for operation in operations:
        operations_by_key[operation["operation_key"]].append(operation)

    all_pairs: list[dict[str, Any]] = []
    group_sizes: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    for subgoal in subgoals:
        task_vector = task_vectors[subgoal_index[subgoal["subgoal_id"]]]

        def features(operation: dict[str, Any]) -> tuple[float, float]:
            cosine = float(task_vector @ operation_vectors[operation_index[operation["operation_id"]]])
            distance = structural_distance(subgoal["signature"], operation["source_signature"])
            return cosine, distance

        same_type = operations_by_key[subgoal["operation_key"]]
        compatible: list[tuple[dict[str, Any], float, float]] = []
        conflicts: list[tuple[dict[str, Any], float, float]] = []
        for operation in same_type:
            cosine, distance = features(operation)
            grade, _ = seed_grade(subgoal, operation)
            target = conflicts if grade == 0 else compatible
            target.append((operation, cosine, distance))
        if not compatible:
            raise ValueError(f"no compatible operation for {subgoal['subgoal_id']}")

        compatible.sort(key=lambda item: (item[2], -item[1], item[0]["operation_id"]))
        anchor = compatible[0]
        selected_ids = {anchor[0]["operation_id"]}
        selected_procedures = {procedure_fingerprint(anchor[0])}
        rows = [pair_row(
            subgoal, anchor[0], grade=2, role="typed_direct_anchor",
            reason="same operation type with closest compatible train-derived structure",
            cosine=anchor[1], distance=anchor[2],
        )]

        # One semantic hard positive and one structural hard positive expose
        # variation within the exact operation type without duplicating the
        # anchor.  Both are useful but less exact than the Grade-2 anchor.
        support_orders = (
            sorted(compatible[1:], key=lambda item: (-item[1], item[2], item[0]["operation_id"])),
            compatible[1:],
        )
        support_roles = ("typed_cosine_hard_support", "typed_structural_support")
        for candidates, role in zip(support_orders, support_roles):
            chosen = next((
                item for item in candidates
                if item[0]["operation_id"] not in selected_ids
                and procedure_fingerprint(item[0]) not in selected_procedures
            ), None)
            if chosen is None:
                continue
            selected_ids.add(chosen[0]["operation_id"])
            selected_procedures.add(procedure_fingerprint(chosen[0]))
            rows.append(pair_row(
                subgoal, chosen[0], grade=1, role=role,
                reason="same required operation is transferable but less structurally exact than anchor",
                cosine=chosen[1], distance=chosen[2],
            ))

        conflicts.sort(key=lambda item: (-item[1], item[2], item[0]["operation_id"]))
        negative = next((item for item in conflicts if item[0]["operation_id"] not in selected_ids), None)
        if negative is not None:
            role = "typed_cosine_hard_conflict"
            reason = "same operation type but explicit applicability conflicts with visible task constraint"
        else:
            different_family = []
            for operation in operations:
                if operation["operation_family"] == subgoal["operation_family"]:
                    continue
                cosine, distance = features(operation)
                different_family.append((operation, cosine, distance))
            different_family.sort(key=lambda item: (-item[1], item[2], item[0]["operation_id"]))
            negative = next(item for item in different_family if item[0]["operation_id"] not in selected_ids)
            role = "cross_type_cosine_hard_negative"
            reason = "high-cosine operation from an unrelated procedural family"
            fallback_counts[subgoal["operation_key"]] += 1
        rows.append(pair_row(
            subgoal, negative[0], grade=0, role=role, reason=reason,
            cosine=negative[1], distance=negative[2],
        ))

        if len(rows) < 2 or len({row["operation_id"] for row in rows}) != len(rows):
            raise ValueError(f"invalid compact group for {subgoal['subgoal_id']}")
        group_sizes[str(len(rows))] += 1
        all_pairs.extend(rows)

    train_pairs = [row for row in all_pairs if row["model_split"] == "internal_train"]
    dev_pairs = [row for row in all_pairs if row["model_split"] == "internal_dev"]
    train_families = {row["family_id"] for row in train_pairs}
    dev_families = {row["family_id"] for row in dev_pairs}
    duplicate_count = len(all_pairs) - len({(row["subgoal_id"], row["operation_id"]) for row in all_pairs})
    if train_families & dev_families:
        raise ValueError("family leakage across train/dev")
    if duplicate_count:
        raise ValueError("duplicate subgoal-operation pairs")

    write_jsonl(args.output_dir / "train_pairs_resolved.jsonl", train_pairs)
    write_jsonl(args.output_dir / "internal_dev_pairs_resolved.jsonl", dev_pairs)
    report = {
        "schema_version": "memp.travelplanner.atomic_typed_hard_pairs.v1",
        "subgoal_count": len(subgoals),
        "operation_count": len(operations),
        "pair_count": len(all_pairs),
        "train_pair_count": len(train_pairs),
        "internal_dev_pair_count": len(dev_pairs),
        "grade_counts_by_split": {
            "internal_train": dict(Counter(str(row["final_grade"]) for row in train_pairs)),
            "internal_dev": dict(Counter(str(row["final_grade"]) for row in dev_pairs)),
        },
        "typed_pair_count": sum(row["typed_candidate"] for row in all_pairs),
        "typed_pair_fraction": sum(row["typed_candidate"] for row in all_pairs) / len(all_pairs),
        "selection_role_counts": dict(Counter(row["selection_role"] for row in all_pairs)),
        "group_size_counts": dict(group_sizes),
        "cross_type_negative_fallbacks_by_subgoal_key": dict(fallback_counts),
        "duplicate_pair_count": duplicate_count,
        "train_family_count": len(train_families),
        "internal_dev_family_count": len(dev_families),
        "cross_split_family_count": 0,
        "judge_called": False,
        "label_confidence_weighting": False,
        "validation_or_test_used": False,
        "integrity": {
            "subgoals": digest(args.subgoals),
            "operations": digest(args.operations),
            "embedding_arrays": digest(args.embedding_arrays),
        },
    }
    report["audit_pass"] = all((
        report["train_pair_count"] >= 15000,
        report["train_pair_count"] <= 20000,
        report["typed_pair_fraction"] >= 0.70,
        report["duplicate_pair_count"] == 0,
        report["cross_split_family_count"] == 0,
        report["validation_or_test_used"] is False,
    ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["audit_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
