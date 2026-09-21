#!/usr/bin/env python3
"""Export an auditable Script bank in the released MemP documents.json shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    documents = []
    for item in bank["records"]:
        documents.append(
            {
                "page_content": item["query"],
                "metadata": {
                    "source": item.get("source"),
                    "query": item["query"],
                    "workflow": item["workflow"],
                    "facts": item.get("facts", {}),
                    "build_policy": "direct",
                    "hit": 0,
                    "success": 0,
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"documents": len(documents), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
