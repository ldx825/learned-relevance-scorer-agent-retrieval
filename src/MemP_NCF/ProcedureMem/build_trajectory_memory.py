#!/usr/bin/env python3
"""Build a frozen Trajectory bank aligned one-to-one with the Script bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memp_repro import atomic_json, sha256_file


HERE = Path(__file__).resolve().parent


def interaction_trajectory(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Keep the task interaction while dropping the repeated generic bootstrap.

    Released ALFWorld trajectories begin with a generic instruction followed by
    an assistant ``OK``.  The paper defines a trajectory as task states,
    actions, and observations, so those two bootstrap messages are not memory.
    """
    start = 2 if len(messages) >= 2 else 0
    result = []
    for message in messages[start:]:
        role = "assistant" if message.get("from") == "gpt" else "environment"
        result.append({"role": role, "content": str(message.get("value", ""))})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=HERE / "Alfworld/alfworld_format_traj.json",
    )
    parser.add_argument("--script-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-size", type=int, default=300)
    args = parser.parse_args()

    source_rows = json.loads(args.trajectories.read_text(encoding="utf-8"))[
        : args.memory_size
    ]
    script_payload = json.loads(args.script_bank.read_text(encoding="utf-8"))
    script_rows = script_payload["records"]
    if len(source_rows) != args.memory_size or len(script_rows) != args.memory_size:
        raise ValueError("trajectory and Script banks must both match --memory-size")

    records = []
    for index, (source, script) in enumerate(zip(source_rows, script_rows, strict=True)):
        query = source["query"].split("\n\n")[0].strip()
        if int(script["source_index"]) != index or script["query"] != query:
            raise ValueError(f"Script/Trajectory alignment mismatch at index {index}")
        records.append(
            {
                "source_index": index,
                "source": source.get("source"),
                "query": query,
                "trajectory": interaction_trajectory(source["trajectory"]),
                # Kept for checkpoint compatibility only.  Trajectory prompts do
                # not expose this Script text to the acting agent.
                "workflow": script["workflow"],
                "facts": source.get("facts"),
            }
        )

    payload = {
        "schema_version": "memp.alfworld_trajectory_bank.v1",
        "trajectory_path": str(args.trajectories.resolve()),
        "trajectory_sha256": sha256_file(args.trajectories),
        "script_bank_path": str(args.script_bank.resolve()),
        "script_bank_sha256": sha256_file(args.script_bank),
        "memory_size": args.memory_size,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "records": len(records),
                "positionwise_query_matches": len(records),
                "unique_queries": len({row["query"] for row in records}),
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
