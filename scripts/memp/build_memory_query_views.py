#!/usr/bin/env python3
"""Expand train-only ALFWorld tasks into query-only Memory retrieval views.

This mirrors the phase expansion used by the ALFWorld SkillDAG+NCF data
pipeline, but never reads a workflow or an official Dev/Test example.  Every
derived text is a deterministic view of a public ALFWorld train query.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VIEW_ORDER = ("full_task", "operation", "object", "destination")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def operation_focus(signature: dict[str, Any]) -> str:
    operation = str(signature["operation"])
    cardinality = int(signature["cardinality"])
    descriptions = {
        "clean": "cleaning one object before placement",
        "heat": "heating one object before placement",
        "cool": "cooling one object before placement",
        "pick_place": "picking up and placing one object",
        "pick_two": "sequentially picking up and placing two objects, one at a time",
        "examine": "examining one object with the required device",
    }
    if operation not in descriptions:
        raise ValueError(f"unsupported operation: {operation}")
    text = descriptions[operation]
    if cardinality == 2 and "two objects" not in text:
        text += " with cardinality two"
    return text


def object_focus(signature: dict[str, Any]) -> str:
    count = "two instances of" if int(signature["cardinality"]) == 2 else "the"
    return f"locating and correctly handling {count} {signature['object_type']}"


def destination_focus(signature: dict[str, Any]) -> str:
    if signature["operation"] == "examine":
        return f"using the {signature['destination']} to examine the target object"
    count = "both objects" if int(signature["cardinality"]) == 2 else "the object"
    return f"navigating to and placing {count} in or on {signature['destination']}"


def view_text(query: str, signature: dict[str, Any], view_type: str) -> str:
    if view_type == "full_task":
        return query
    focus = {
        "operation": operation_focus,
        "object": object_focus,
        "destination": destination_focus,
    }[view_type](signature)
    # Keep the full train task in every focused view, exactly as the SkillDAG
    # phase context keeps the full task alongside the current phase.
    return f"Full task: {query}\nCurrent retrieval focus: {focus}."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_jsonl(args.dataset / "tasks.jsonl")
    views: list[dict[str, Any]] = []
    for task in tasks:
        for view_index, view_type in enumerate(VIEW_ORDER):
            views.append(
                {
                    "schema_version": "task_resource_ncf.memory_query_view.v1",
                    "query_view_id": f"task-{int(task['task_id']):04d}::{view_type}",
                    "query_view_index": len(views),
                    "parent_task_id": int(task["task_id"]),
                    "view_index": view_index,
                    "view_type": view_type,
                    "task_query": task["query"],
                    "query_text": view_text(task["query"], task["signature"], view_type),
                    "signature": task["signature"],
                    "canonical_id": task["canonical_id"],
                    "model_split": task["model_split"],
                    "source": "deterministic_view_of_public_alfworld_train_query",
                    "uses_workflow_text": False,
                    "uses_eval_data": False,
                }
            )

    split_counts = Counter(row["model_split"] for row in views)
    view_counts = Counter(row["view_type"] for row in views)
    report = {
        "schema_version": "task_resource_ncf.memory_query_views_report.v1",
        "parent_train_query_count": len(tasks),
        "query_view_count": len(views),
        "views_per_parent": len(VIEW_ORDER),
        "view_counts": dict(sorted(view_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "model_inputs": ["task_query", "memory_source_query"],
        "workflow_text_used": False,
        "official_dev_or_test_read": False,
        "api_calls_made": 0,
    }
    write_jsonl(args.output_dir / "query_views.jsonl", views)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
