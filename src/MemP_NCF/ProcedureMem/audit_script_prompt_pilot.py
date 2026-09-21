#!/usr/bin/env python3
"""Compare Script banks built with different prompt reconstruction modes."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


TERM_GROUPS = {
    "explicit_preconditions": ("precondition", "must", "ensure", "required"),
    "appliance_hints": ("microwave", "fridge", "sink basin", "sinkbasin", "refrigerator"),
    "instance_specific": (" 1", " 2", " 3", " 4"),
    "travelplanner_leakage": (
        "flightsearch",
        "accommodationsearch",
        "restaurantsearch",
        "attractionsearch",
        "planner tool",
    ),
}


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", payload)
    return list(records[:limit])


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(item["workflow"].split()) for item in records]
    result: dict[str, Any] = {
        "records": len(records),
        "mean_words": round(statistics.mean(lengths), 2),
        "median_words": round(statistics.median(lengths), 2),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "under_20_words": sum(length < 20 for length in lengths),
        "over_60_words": sum(length > 60 for length in lengths),
        "list_format_violations": sum(
            bool(re.search(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+", item["workflow"]))
            for item in records
        ),
        "task_operation_misses": 0,
    }
    operation_terms = {
        "clean": ("clean",),
        "heat": ("heat", "hot"),
        "cool": ("cool", "cold"),
        "look": ("look", "examine", "use", "toggle"),
        "examine": ("look", "examine", "use", "toggle"),
    }
    for item in records:
        query = item["query"].lower()
        workflow = item["workflow"].lower()
        required = [terms for operation, terms in operation_terms.items() if operation in query]
        if "put" in query:
            required.append(("put", "place"))
        if any(not any(term in workflow for term in terms) for terms in required):
            result["task_operation_misses"] += 1
    for group, terms in TERM_GROUPS.items():
        result[group] = sum(
            any(term in item["workflow"].lower() for term in terms)
            for item in records
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhanced", type=Path, required=True)
    parser.add_argument("--paper-minimal", type=Path, required=True)
    parser.add_argument("--paper-minimal-v2", type=Path, required=True)
    parser.add_argument("--paper-minimal-v3", type=Path, required=True)
    parser.add_argument("--release-exact", type=Path, required=True)
    parser.add_argument("--release-corrected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    banks = {
        "enhanced": load_records(args.enhanced, args.limit),
        "paper_minimal": load_records(args.paper_minimal, args.limit),
        "paper_minimal_v2": load_records(args.paper_minimal_v2, args.limit),
        "paper_minimal_v3": load_records(args.paper_minimal_v3, args.limit),
        "release_exact": load_records(args.release_exact, args.limit),
        "release_corrected": load_records(args.release_corrected, args.limit),
    }
    source_indices = {
        name: [int(item["source_index"]) for item in records]
        for name, records in banks.items()
    }
    if len({tuple(indices) for indices in source_indices.values()}) != 1:
        raise RuntimeError(f"banks do not contain the same source indices: {source_indices}")

    report = {
        "limit": args.limit,
        "source_indices": source_indices["enhanced"],
        "metrics": {name: summarize(records) for name, records in banks.items()},
        "examples": [
            {
                "source_index": source_index,
                "query": banks["enhanced"][offset]["query"],
                "enhanced": banks["enhanced"][offset]["workflow"],
                "paper_minimal": banks["paper_minimal"][offset]["workflow"],
                "paper_minimal_v2": banks["paper_minimal_v2"][offset]["workflow"],
                "paper_minimal_v3": banks["paper_minimal_v3"][offset]["workflow"],
                "release_exact": banks["release_exact"][offset]["workflow"],
                "release_corrected": banks["release_corrected"][offset]["workflow"],
            }
            for offset, source_index in enumerate(source_indices["enhanced"])
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# MemP Script prompt pilot audit",
        "",
        "All modes use the same first 30 public gold trajectories.",
        "",
        "| mode | mean words | median | <20 | >60 | list violations | operation misses | precondition terms | appliance hints | TravelPlanner leakage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["metrics"].items():
        lines.append(
            f"| {name} | {metrics['mean_words']} | {metrics['median_words']} | "
            f"{metrics['under_20_words']} | {metrics['over_60_words']} | "
            f"{metrics['list_format_violations']} | {metrics['task_operation_misses']} | "
            f"{metrics['explicit_preconditions']} | {metrics['appliance_hints']} | "
            f"{metrics['travelplanner_leakage']} |"
        )
    lines.extend(["", "## Side-by-side examples", ""])
    for example in report["examples"]:
        lines.extend(
            [
                f"### source {example['source_index']}: {example['query']}",
                "",
                f"- enhanced: {example['enhanced']}",
                "",
                f"- paper-minimal: {example['paper_minimal']}",
                "",
                f"- paper-minimal-v2: {example['paper_minimal_v2']}",
                "",
                f"- paper-minimal-v3: {example['paper_minimal_v3']}",
                "",
                f"- release-exact: {example['release_exact']}",
                "",
                f"- release-corrected: {example['release_corrected']}",
                "",
            ]
        )
    (args.output_dir / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
