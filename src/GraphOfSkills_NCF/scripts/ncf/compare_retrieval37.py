#!/usr/bin/env python3
"""Zero-API comparison on held-out ALFWorld tasks."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

from gos.ncf.retrieval37 import ALFWorld37Retriever


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()

    archive = root / "data/alfworld_task_skill"
    shared = archive / "shared"
    retriever = ALFWorld37Retriever(
        graph_path=shared / "skillgraph_alfworld.json",
        embeddings_path=shared / "embeddings/skill_embeddings_37.json",
        model_path=root
        / ".runtime/gos_ncf/models/neumf_graded_v2_clean/neumf_graded.pt",
    )

    splits = json.loads(
        (shared / "splits/task_level_700.json").read_text()
    )["record_ids"]
    selected = set(splits[args.split])
    tasks = {
        row["record_id"]: row
        for row in load_jsonl(shared / "raw/tasks_all_official_splits.jsonl")
        if row["record_id"] in selected
    }
    embeddings: dict[str, Any] = {}
    for filename in glob.glob(
        str(shared / "embeddings/task_embeddings_official_train/shard_*.json")
    ):
        embeddings.update(json.loads(Path(filename).read_text()))
    labels: dict[str, dict[str, int]] = {}
    for row in load_jsonl(
        root
        / "data/alfworld_task_skill/task_skill_v2_gos_clean/labels/judge_clean.jsonl"
    ):
        if row["task_record_id"] in selected and row.get("trainable"):
            labels.setdefault(row["task_record_id"], {})[row["skill_id"]] = int(
                row["target_grade"]
            )

    details: list[dict[str, Any]] = []
    summary: dict[str, dict[str, list[float]]] = {
        method: {} for method in ("graph", "graph_ncf")
    }
    for task_id in sorted(selected):
        embedding = embeddings[task_id]["embedding"]
        judged = labels.get(task_id, {})
        ideal = sorted(judged.values(), reverse=True)[: args.top_k]
        required_total = sum(grade == 2 for grade in judged.values())
        task_detail = {
            "task_record_id": task_id,
            "task_type": tasks[task_id]["task_type"],
            "task_text": tasks[task_id]["task_text"],
            "methods": {},
        }
        for method in ("graph", "graph_ncf"):
            rows = retriever.retrieve(embedding, method=method, top_k=args.top_k)
            grades = [judged.get(row.skill_id, -1) for row in rows]
            known = [grade for grade in grades if grade >= 0]
            metrics = {
                "ndcg_known@3": (
                    dcg([max(grade, 0) for grade in grades]) / dcg(ideal)
                    if dcg(ideal)
                    else 0.0
                ),
                "required_recall@3": (
                    sum(grade == 2 for grade in grades) / required_total
                    if required_total
                    else 0.0
                ),
                "required_hit@3": float(any(grade == 2 for grade in grades)),
                "bad_rate_known@3": (
                    sum(grade == 0 for grade in known) / len(known) if known else 0.0
                ),
                "judged_coverage@3": len(known) / len(grades),
            }
            for name, value in metrics.items():
                summary[method].setdefault(name, []).append(float(value))
            task_detail["methods"][method] = {
                "skills": [
                    {
                        "skill_id": row.skill_id,
                        "grade": grade,
                        "score": row.score,
                        "graph_score": row.graph_score,
                        "ncf_score": row.ncf_score,
                    }
                    for row, grade in zip(rows, grades)
                ],
                "metrics": metrics,
            }
        details.append(task_detail)

    report = {
        "schema_version": "gos_ncf.retrieval37_comparison.v1",
        "split": args.split,
        "task_count": len(details),
        "top_k": args.top_k,
        "api_calls_made": 0,
        "methods": {
            method: {
                name: sum(values) / len(values)
                for name, values in metrics.items()
            }
            for method, metrics in summary.items()
        },
        "notes": [
            "Both methods use the same 37 skills, frozen embeddings, and frozen graph.",
            "graph_ncf reranks the top-12 graph candidates; graph returns PageRank order.",
            "Only Judge-labeled pairs contribute known grades; judged_coverage reports label coverage.",
        ],
        "details": details,
    }
    output = (
        root
        / f".runtime/gos_ncf/reports/retrieval37_{args.split}_comparison.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({**report, "details": f"{len(details)} task rows omitted"}, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
