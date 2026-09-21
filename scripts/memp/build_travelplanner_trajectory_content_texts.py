#!/usr/bin/env python3
"""Build retrieval content texts for the Trajectory/Proceduralization banks.

For each memory document we render the PROCEDURAL CONTENT (the trajectory
workflow steps, plus the proceduralization script when present) into a single
text used for embedding and for prompt rendering.  The source task query stays
a provenance label only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render_steps(steps: list[dict[str, Any]]) -> str:
    lines = []
    for step in steps:
        parts = [f"Step {step.get('step', '?')}:"]
        for field in ("thought", "action", "observation"):
            value = str(step.get(field) or "").strip()
            if value:
                parts.append(value)
        lines.append(" ".join(parts))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for format_name in ("trajectory", "proceduralization"):
        documents = json.loads(
            (args.bank_root / format_name / "documents.json").read_text(encoding="utf-8")
        )
        for index, document in enumerate(documents):
            metadata = document.get("metadata", {})
            workflow = metadata.get("workflow", {})
            chunks: list[str] = []
            if isinstance(workflow, dict):
                script = str(workflow.get("script") or "").strip()
                if script:
                    chunks.append("Proceduralized plan:\n" + script)
                steps = workflow.get("trajectory")
            else:
                steps = workflow
            if isinstance(steps, list) and steps:
                chunks.append("Executed trajectory:\n" + render_steps(steps))
            content = "\n\n".join(chunks).strip()
            if not content:
                content = str(document.get("page_content") or "").strip()
            rows.append({
                "format": format_name,
                "memory_id": f"{format_name}_{index:03d}",
                "source": metadata.get("source"),
                "source_query": str(metadata.get("query") or "").strip(),
                "content_text": content,
                "validation_or_test_used": False,
            })
    if len(rows) != 46:
        raise ValueError(f"expected 46 documents, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    for row in rows[:1]:
        print(f"[{row['format']}] {row['source']}: {len(row['content_text'])} chars")
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
