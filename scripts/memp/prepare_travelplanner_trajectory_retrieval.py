#!/usr/bin/env python3
"""Generate task-level retrieval JSONs for the Trajectory/Proceduralization banks.

The retrieval unit is the full task query (not the five-phase decomposition):
cosine or the frozen V25 Content-NeuMF scores each memory document's PROCEDURAL
CONTENT text, the Top-K documents are rendered (truncated) into
`dynamic_memory_text` for the existing online runner.  The source task query is
kept only as a provenance label.  No validation/test reward, trajectory or gold
plan is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NCF_SRC = ROOT / "src/GraphOfSkills_NCF"
for value in (str(HERE), str(NCF_SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from build_memory_embedding_candidates import embed_texts  # noqa: E402
from gos.ncf.models import NCFConfig, NeuMF  # noqa: E402


MAX_RENDER_CHARS = 4000
MAX_EMBED_CHARS = 8000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_dynamic_memory(task_query: str, selected: list[dict[str, Any]]) -> str:
    lines = [
        "Retrieved procedural memories for the current task:",
        f"Current task: {task_query}",
        "",
    ]
    for rank, memory in enumerate(selected, 1):
        content = str(memory["content_text"])
        if len(content) > MAX_RENDER_CHARS:
            content = content[:MAX_RENDER_CHARS] + "\n...[truncated]"
        lines.append(
            f"--- Memory {rank} (source task: {memory.get('source_query') or memory.get('source')}) ---"
        )
        lines.append(content)
    lines.append("")
    lines.append(
        "Evidence policy: keep exact tool-returned entity names, cities, dates, "
        "prices and identifiers for the CURRENT task; do not copy source-task "
        "values or invent evidence."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-texts", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--format", choices=("trajectory", "proceduralization", "both"), required=True)
    parser.add_argument("--reranker", choices=("cosine", "neumf"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--quality-bank",
        type=Path,
        help="V26 41-document quality bank jsonl; overrides --content-texts",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.reranker == "neumf" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for NeuMF")
    if args.quality_bank is not None:
        documents = read_jsonl(args.quality_bank)
        if len(documents) not in (41, 84):
            raise ValueError(f"expected 41 or 84 quality-bank documents, got {len(documents)}")
        for row in documents:
            row["format"] = "trajectory"
    else:
        rows = read_jsonl(args.content_texts)
        if len(rows) != 46:
            raise ValueError("expected 46 content documents")
        if any(row.get("validation_or_test_used") is not False for row in rows):
            raise ValueError("evaluation-derived memory content detected")
        formats = ("trajectory", "proceduralization") if args.format == "both" else (args.format,)
        documents = [row for row in rows if row["format"] in formats]
        if not documents:
            raise ValueError(f"no documents for format={args.format}")

    with args.csv.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    indices = list(range(args.start, min(args.start + args.count, len(table))))
    if not indices:
        raise ValueError("empty task slice")
    task_queries = [str(table[index]["query"]) for index in indices]

    texts = [str(row["content_text"])[:MAX_EMBED_CHARS] for row in documents] + task_queries
    vectors = embed_texts(
        texts,
        model="text-embedding-3-small",
        cache_path=args.cache_dir / "trajectory_content_embeddings.json",
        api_base=os.environ.get("SKILLDAG_EMBEDDING_BASE", "https://yunwu.ai/v1"),
        api_key=os.environ.get("SKILLDAG_EMBEDDING_API_KEY", ""),
        batch_size=64,
        timeout=180,
    ).astype(np.float32)
    doc_vectors = vectors[: len(documents)]
    query_vectors = vectors[len(documents):]
    doc_norm = doc_vectors / np.maximum(np.linalg.norm(doc_vectors, axis=1, keepdims=True), 1e-12)
    query_norm = query_vectors / np.maximum(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12)
    cosine = query_norm @ doc_norm.T

    if args.reranker == "neumf":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = NeuMF(NCFConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scores = np.empty((len(query_vectors), len(documents)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(query_vectors), 16):
                batch = query_vectors[start : start + 16]
                task_batch = np.repeat(batch[:, None, :], len(documents), axis=1).reshape(-1, batch.shape[1])
                memory_batch = np.tile(doc_vectors, (len(batch), 1))
                output = model(torch.from_numpy(task_batch), torch.from_numpy(memory_batch)).numpy()
                scores[start : start + len(batch)] = output.reshape(len(batch), len(documents))
    else:
        scores = cosine

    retrievals = []
    for local_index, task_index in enumerate(indices):
        order = np.argsort(-scores[local_index], kind="stable")[: args.top_k]
        selected = []
        for rank, document_index in enumerate(order, 1):
            document = documents[int(document_index)]
            selected.append({
                "rank": rank,
                "format": document["format"],
                "memory_id": document["memory_id"],
                "source": document["source"],
                "source_query": document["source_query"],
                "score": float(scores[local_index][document_index]),
                "cosine_score": float(cosine[local_index][document_index]),
                "content_text": document["content_text"],
            })
        retrievals.append({
            "test_index": task_index,
            "query": task_queries[local_index],
            "dynamic_memory_text": render_dynamic_memory(task_queries[local_index], selected),
            "selected_memories": [
                {
                    "rank": item["rank"],
                    "format": item["format"],
                    "memory_id": item["memory_id"],
                    "source": item["source"],
                    "score": item["score"],
                    "cosine_score": item["cosine_score"],
                }
                for item in selected
            ],
            "retriever": args.reranker,
            "memory_format": args.format,
            "top_k": args.top_k,
            "gold_reward_or_trajectory_used": False,
        })

    payload = {
        "schema_version": "memp.travelplanner.trajectory_content_retrieval.v1",
        "split": args.split,
        "task_slice": [min(indices), max(indices) + 1],
        "selected_indices": indices,
        "task_count": len(indices),
        "memory_format": args.format,
        "memory_document_count": len(documents),
        "top_k": args.top_k,
        "reranker": args.reranker,
        "score_fusion": False,
        "query_unit": "full_task_query",
        "memory_unit": "procedural_content_text",
        "official_gold_reward_or_trajectory_used": False,
        "retrievals": retrievals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "retrievals"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
