#!/usr/bin/env python3
"""Derive binary-label and random-negative ALFWorld ablation arrays.

Only the intended factor changes.  Embeddings, query/resource order, split
assignments, positive rows, pair counts, and per-query negative counts remain
frozen.  No API or official evaluation data is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/ablations/alfworld_ncf_v1.json"
DEFAULT_OUTPUT = ROOT / ".runtime/ablations/alfworld_ncf_v1/derived_arrays"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_variant(
    *,
    source_path: Path,
    output_dir: Path,
    domain: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}

    if domain == "skill":
        query_key = "pair_phase_indices"
        resource_key = "pair_skill_indices"
        resource_ids_key = "skill_ids"
    elif domain == "memory":
        query_key = "pair_task_indices"
        resource_key = "pair_memory_indices"
        resource_ids_key = "memory_ids"
    else:
        raise ValueError(f"unsupported domain: {domain}")

    original_grade = arrays["target_grade"].copy()
    original_relevance = arrays["target_relevance"].copy()
    original_resources = arrays[resource_key].copy()
    query_indices = arrays[query_key]
    resource_count = len(arrays[resource_ids_key])
    rng = np.random.default_rng(seed)
    changed_resource_rows = 0

    if variant == "binary":
        # Keep the original 2/1/0 grade for internal-dev model selection and
        # final graded evaluation.  Only the pointwise training target loses
        # the Grade-2 versus Grade-1 distinction.
        arrays["target_relevance"] = (original_grade > 0).astype(np.float32)
    elif variant == "random_negative":
        for query_index in np.unique(query_indices):
            positions = np.flatnonzero(query_indices == query_index)
            positive_positions = positions[original_grade[positions] > 0]
            negative_positions = positions[original_grade[positions] == 0]
            known_positive_resources = set(int(value) for value in original_resources[positive_positions])
            eligible = np.asarray(
                [index for index in range(resource_count) if index not in known_positive_resources],
                dtype=np.int64,
            )
            if len(negative_positions) > len(eligible):
                raise ValueError(
                    f"query {query_index} needs {len(negative_positions)} random negatives "
                    f"but only {len(eligible)} resources are eligible"
                )
            replacements = rng.choice(eligible, size=len(negative_positions), replace=False)
            changed_resource_rows += int(np.sum(arrays[resource_key][negative_positions] != replacements))
            arrays[resource_key][negative_positions] = replacements.astype(arrays[resource_key].dtype)
            for position, resource_index in zip(negative_positions, replacements, strict=True):
                arrays["pair_ids"][position] = (
                    f"ablation-random-negative::{domain}::{int(query_index)}::{int(resource_index)}"
                )
    elif variant == "random_negative_matched":
        # Randomize only the task-resource association of *training* negatives.
        # Starting from the original valid assignment and swapping resources
        # between two rows preserves exactly:
        #   - the global negative-resource frequency distribution,
        #   - the number of negatives per query,
        #   - labels, weights, splits, and all dev/test rows.
        # A proposed swap is accepted only when it introduces neither a known
        # positive nor a duplicate resource for either query.
        train_negative_positions = np.flatnonzero(
            (arrays["split_codes"] == 0) & (original_grade == 0)
        )
        positive_by_query: dict[int, set[int]] = {}
        negative_by_query: dict[int, set[int]] = {}
        for position in range(len(query_indices)):
            query_index = int(query_indices[position])
            resource_index = int(arrays[resource_key][position])
            if int(original_grade[position]) > 0:
                positive_by_query.setdefault(query_index, set()).add(resource_index)
            elif int(arrays["split_codes"][position]) == 0:
                negative_by_query.setdefault(query_index, set()).add(resource_index)

        successful_swaps = 0
        target_swaps = 10 * len(train_negative_positions)
        max_attempts = 100 * len(train_negative_positions)
        for _ in range(max_attempts):
            if successful_swaps >= target_swaps:
                break
            left, right = rng.choice(train_negative_positions, size=2, replace=False)
            left = int(left)
            right = int(right)
            left_query = int(query_indices[left])
            right_query = int(query_indices[right])
            if left_query == right_query:
                continue
            left_resource = int(arrays[resource_key][left])
            right_resource = int(arrays[resource_key][right])
            if left_resource == right_resource:
                continue
            if right_resource in positive_by_query.get(left_query, set()):
                continue
            if left_resource in positive_by_query.get(right_query, set()):
                continue
            if right_resource in negative_by_query[left_query]:
                continue
            if left_resource in negative_by_query[right_query]:
                continue

            arrays[resource_key][left] = right_resource
            arrays[resource_key][right] = left_resource
            negative_by_query[left_query].remove(left_resource)
            negative_by_query[left_query].add(right_resource)
            negative_by_query[right_query].remove(right_resource)
            negative_by_query[right_query].add(left_resource)
            successful_swaps += 1

        if successful_swaps < target_swaps:
            raise RuntimeError(
                f"controlled randomization stopped after {successful_swaps} successful "
                f"swaps; expected {target_swaps}"
            )
        changed_mask = arrays[resource_key] != original_resources
        changed_resource_rows = int(np.sum(changed_mask))
        for position in train_negative_positions:
            resource_index = int(arrays[resource_key][position])
            arrays["pair_ids"][position] = (
                f"ablation-random-negative-matched::{domain}::"
                f"{int(query_indices[position])}::{resource_index}"
            )
    else:
        raise ValueError(f"unsupported variant: {variant}")

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)

    # Fail closed on every invariant not targeted by the ablation.
    with np.load(arrays_path, allow_pickle=False) as replay:
        if len(replay["pair_ids"]) != len(original_grade):
            raise AssertionError("pair count changed")
        if not np.array_equal(replay["split_codes"], arrays["split_codes"]):
            raise AssertionError("split assignments changed")
        if variant == "binary":
            if not np.array_equal(replay[resource_key], original_resources):
                raise AssertionError("binary ablation changed candidate identities")
            if not np.array_equal(replay["target_grade"], original_grade):
                raise AssertionError("binary ablation changed evaluation grades")
            if not np.array_equal(
                replay["target_relevance"], (original_grade > 0).astype(np.float32)
            ):
                raise AssertionError("binary relevance conversion failed")
        else:
            positive_mask = original_grade > 0
            if not np.array_equal(replay[resource_key][positive_mask], original_resources[positive_mask]):
                raise AssertionError("random-negative ablation changed positive rows")
            if not np.array_equal(replay["target_grade"], original_grade):
                raise AssertionError("random-negative ablation changed labels")
            if not np.array_equal(replay["target_relevance"], original_relevance):
                raise AssertionError("random-negative ablation changed relevance targets")
            if variant == "random_negative_matched":
                train_negative_mask = (arrays["split_codes"] == 0) & (original_grade == 0)
                frozen_mask = ~train_negative_mask
                if not np.array_equal(
                    replay[resource_key][frozen_mask], original_resources[frozen_mask]
                ):
                    raise AssertionError("matched random negatives changed non-training rows")
                before_counts = np.bincount(
                    original_resources[train_negative_mask], minlength=resource_count
                )
                after_counts = np.bincount(
                    replay[resource_key][train_negative_mask], minlength=resource_count
                )
                if not np.array_equal(before_counts, after_counts):
                    raise AssertionError("matched random negatives changed resource frequencies")

    report = {
        "schema_version": "agent_skill_evolution.ncf_training_variant.v1",
        "domain": domain,
        "variant": variant,
        "seed": seed,
        "source_arrays": str(source_path.resolve()),
        "source_sha256": sha256(source_path),
        "output_arrays": str(arrays_path.resolve()),
        "output_sha256": sha256(arrays_path),
        "pair_count": int(len(original_grade)),
        "grade_counts_before": {
            str(grade): int(np.sum(original_grade == grade)) for grade in (0, 1, 2)
        },
        "grade_counts_after": {
            str(grade): int(np.sum(arrays["target_grade"] == grade)) for grade in (0, 1, 2)
        },
        "changed_resource_rows": changed_resource_rows,
        "official_eval_data_read": False,
    }
    (output_dir / "variant_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--domains", nargs="+", choices=("skill", "memory"), default=("skill", "memory")
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("binary", "random_negative", "random_negative_matched"),
        default=("binary", "random_negative", "random_negative_matched"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reports = []
    for domain, section in (("skill", "skill_alfworld"), ("memory", "memory_alfworld")):
        if domain not in args.domains:
            continue
        source = ROOT / manifest[section]["arrays"]["path"]
        for variant in args.variants:
            reports.append(
                save_variant(
                    source_path=source,
                    output_dir=args.output_root / domain / str(args.seed) / variant,
                    domain=domain,
                    variant=variant,
                    seed=args.seed,
                )
            )
    print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
