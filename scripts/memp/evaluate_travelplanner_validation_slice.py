#!/usr/bin/env python3
"""Evaluate a completed TravelPlanner validation prefix with official constraints."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


CS_KEYS = (
    "is_reasonable_visiting_city",
    "is_valid_restaurants",
    "is_valid_attractions",
    "is_valid_accommodation",
    "is_valid_transportation",
    "is_valid_information_in_current_city",
    "is_valid_information_in_sandbox",
    "is_not_absent",
)
HC_TO_QUERY_KEY = {
    "valid_room_rule": "house rule",
    "valid_cuisine": "cuisine",
    "valid_room_type": "room type",
    "valid_transportation": "transportation",
}


def parse_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = dict(row)
    for key in ("days", "visiting_city_number", "people_number", "budget"):
        parsed[key] = int(parsed[key])
    parsed["date"] = ast.literal_eval(parsed["date"])
    parsed["local_constraint"] = ast.literal_eval(parsed["local_constraint"])
    return parsed


def normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--indices",
        help="Optional comma-separated validation indices; overrides --limit",
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=Path(".runtime/memp/travelplanner/hf_dataset_repo/validation.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(".runtime/memp/travelplanner/script_query_cosine_top10_validation30/results"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    csv_path = args.validation_csv if args.validation_csv.is_absolute() else root / args.validation_csv
    results_dir = args.results_dir if args.results_dir.is_absolute() else root / args.results_dir
    with csv_path.open(encoding="utf-8", newline="") as handle:
        all_rows = [parse_row(row) for row in csv.DictReader(handle)]
    if args.indices:
        selected_indices = [
            int(value.strip()) for value in args.indices.split(",") if value.strip()
        ]
        if not selected_indices or len(selected_indices) != len(set(selected_indices)):
            raise ValueError("--indices must contain unique validation indices")
        if any(index < 0 or index >= len(all_rows) for index in selected_indices):
            raise ValueError(f"validation indices must be in [0, {len(all_rows)})")
    else:
        if not 0 < args.limit <= len(all_rows):
            raise ValueError(f"--limit must be in [1, {len(all_rows)}]")
        selected_indices = list(range(args.limit))
    rows = [all_rows[index] for index in selected_indices]

    tp_root = root / "references/TravelPlanner"
    evaluation_dir = tp_root / "evaluation"
    previous_cwd = Path.cwd()
    os.chdir(evaluation_dir)
    sys.path.insert(0, str(evaluation_dir))
    sys.path.insert(0, str(tp_root))
    if "gradio" not in sys.modules:
        gradio_stub = types.ModuleType("gradio")
        gradio_stub.Error = RuntimeError
        sys.modules["gradio"] = gradio_stub
    try:
        from commonsense_constraint import evaluation as commonsense_eval
        from hard_constraint import evaluation as hard_eval

        details: list[dict[str, Any]] = []
        cs_pass = 0
        hc_pass = 0
        hc_evaluated_total = 0
        hc_evaluated_tasks = 0
        delivered = 0
        all_pass = 0
        executed_steps = 0
        completed = 0
        for validation_index, query in zip(selected_indices, rows, strict=True):
            path = results_dir / f"test_{validation_index:04d}.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            completed += 1
            executed_steps += int(record.get("executed_steps", 0))
            plan = record.get("postprocess", {}).get("parsed_plan") or []
            if plan:
                delivered += 1
                cs = commonsense_eval(query, plan)
            else:
                cs = None

            cs_values = {
                key: bool(cs and cs.get(key, (False, None))[0]) for key in CS_KEYS
            }
            cs_pass += sum(cs_values.values())
            gate = bool(
                cs
                and cs.get("is_not_absent", (False, None))[0]
                and cs.get("is_valid_information_in_sandbox", (False, None))[0]
            )
            hard = hard_eval(query, plan) if gate else None
            hc_evaluated_tasks += int(gate)
            applicable = {"valid_cost": True}
            applicable.update(
                {
                    metric: query["local_constraint"].get(query_key) is not None
                    for metric, query_key in HC_TO_QUERY_KEY.items()
                }
            )
            hard_values = {
                metric: bool(hard and hard.get(metric, (False, None))[0])
                for metric, is_applicable in applicable.items()
                if is_applicable
            }
            hc_pass += sum(hard_values.values())
            hc_evaluated_total += len(hard_values) if gate else 0
            task_pass = all(cs_values.values()) and all(hard_values.values())
            all_pass += int(task_pass)
            details.append(
                {
                    "validation_index": validation_index,
                    "delivered": bool(plan),
                    "executed_steps": int(record.get("executed_steps", 0)),
                    "commonsense": normalize(cs),
                    "hard": normalize(hard),
                    "task_pass": task_pass,
                }
            )
    finally:
        os.chdir(previous_cwd)

    # Match the paper's official aggregation: every task contributes one
    # budget constraint, and each explicitly requested local constraint adds
    # one more denominator item. A task blocked by the commonsense sandbox
    # gate contributes no hard-constraint passes, but remains in the fixed
    # denominator (TravelPlanner's full validation denominator is 420).
    paper_hc_total = len(rows) + sum(
        sum(value is not None for value in query["local_constraint"].values())
        for query in rows
    )
    report = {
        "protocol": "TravelPlanner official constraint functions on a validation selection",
        "selected_indices": selected_indices,
        "requested": len(rows),
        "completed": completed,
        "delivery_rate": delivered / completed if completed else None,
        "commonsense_micro_pass_rate": cs_pass / (completed * len(CS_KEYS)) if completed else None,
        "commonsense_pass": cs_pass,
        "commonsense_total": completed * len(CS_KEYS),
        "hard_micro_pass_rate": hc_pass / paper_hc_total if paper_hc_total else None,
        "hard_pass": hc_pass,
        "hard_total": paper_hc_total,
        "hard_evaluated_task_count": hc_evaluated_tasks,
        "hard_evaluated_applicable_total": hc_evaluated_total,
        "final_pass_rate": all_pass / completed if completed else None,
        "final_pass": all_pass,
        "avg_steps": executed_steps / completed if completed else None,
        "partial_prefix_metric": selected_indices != list(range(len(all_rows))),
        "details": details,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
