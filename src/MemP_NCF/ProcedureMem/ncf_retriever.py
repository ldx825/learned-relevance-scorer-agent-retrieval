"""Content-NeuMF residual reranker for the frozen MemP Script bank."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
NCF_SRC = ROOT / "src/GraphOfSkills_NCF"
MEMP_SCRIPTS = ROOT / "scripts/memp"
if str(NCF_SRC) not in sys.path:
    sys.path.insert(0, str(NCF_SRC))
if str(MEMP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MEMP_SCRIPTS))

from gos.ncf.models import NCFConfig, NeuMF  # noqa: E402
from memory_task_schema import parse_query, script_validity  # noqa: E402
from memory_ncf_model import (  # noqa: E402
    PAIR_FEATURE_NAMES,
    RelationNeuMF,
    relation_features,
)


def _text_key(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def memory_content_text(row: dict[str, Any]) -> str:
    return f"Memory task: {row['query']}\nReusable procedure: {row['workflow']}"


def structured_features(query: str, config: dict[str, Any]) -> np.ndarray:
    signature = parse_query(query).as_dict()
    vocab = config["vocab"]
    scale = float(config["scale"])
    values = []
    for field in ("operation", "object_type", "cardinality", "destination"):
        choices = vocab[field]
        value = str(signature[field])
        vector = np.zeros(len(choices), dtype=np.float32)
        vector[choices.index(value) if value in choices else choices.index("<UNK>")] = scale
        values.append(vector)
    return np.concatenate(values)


class MemoryNCFReranker:
    """Rerank a Query-cosine candidate set with a frozen Content-NeuMF.

    Legacy residual checkpoints keep their original bounded-correction path.
    Query-only V17 checkpoints use a pure two-stage route: cosine Top-C followed
    by direct NeuMF ranking, with no score blending or protected anchors.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        memory_embedding_cache: Path,
        bank: list[dict[str, Any]],
        *,
        embedding_model: str = "text-embedding-3-large",
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        schema = checkpoint.get("schema_version")
        residual_schema = "task_resource_ncf.memory_cosine_residual_neumf_checkpoint.v1"
        query_only_schema = "skilldag_ncf.v3.phase_neumf_checkpoint.v1"
        if schema not in {residual_schema, query_only_schema}:
            raise ValueError(f"unsupported NCF checkpoint schema: {schema!r}")
        self.ranking_mode = "residual_blend" if schema == residual_schema else "direct_neumf"
        config = NCFConfig(**checkpoint["model_config"])
        self.pair_feature_names = (
            checkpoint.get("pair_structured_feature_names") or []
            if self.ranking_mode == "residual_blend"
            else []
        )
        if self.pair_feature_names and self.pair_feature_names != list(PAIR_FEATURE_NAMES):
            raise ValueError("unsupported pair structured feature schema")
        self.model = (
            RelationNeuMF(config, len(self.pair_feature_names))
            if self.pair_feature_names
            else NeuMF(config)
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.correction_scale = (
            float(checkpoint["correction_scale"])
            if self.ranking_mode == "residual_blend"
            else None
        )
        self.embedding_model = embedding_model
        self.input_projection = checkpoint.get("input_projection")
        self.structured_feature_config = checkpoint.get("structured_features")
        self.preserve_query_cosine_top = int(
            checkpoint.get("preserve_query_cosine_top", 0)
        )
        self.memory_signatures = [parse_query(row["query"]).as_dict() for row in bank]
        self.memory_validity = [
            script_validity(parse_query(row["query"]), row["workflow"])[0] for row in bank
        ]

        payload = json.loads(memory_embedding_cache.read_text(encoding="utf-8"))
        if payload.get("model") != embedding_model:
            raise ValueError("NCF memory embedding model mismatch")
        if self.ranking_mode == "direct_neumf":
            if payload.get("queries") != [row["query"] for row in bank]:
                raise ValueError("query-only NCF Memory ordering differs from the frozen bank")
            values = payload.get("embeddings") or []
        else:
            records = payload.get("records") or {}
            values = []
            for row in bank:
                text = memory_content_text(row)
                record = records.get(_text_key(text))
                if not record or record.get("text") != text:
                    raise ValueError(f"missing NCF content embedding for memory {row.get('source_index')}")
                values.append(record["embedding"])
        self.memory_embeddings = np.asarray(values, dtype=np.float32)
        if self.input_projection is not None:
            mean = self.input_projection["mean"].cpu().numpy()
            components = self.input_projection["components"].cpu().numpy()
            self.memory_embeddings = (self.memory_embeddings - mean) @ components
        if self.structured_feature_config is not None:
            memory_features = np.stack(
                [structured_features(row["query"], self.structured_feature_config) for row in bank]
            )
            self.memory_embeddings = np.concatenate(
                (self.memory_embeddings, memory_features), axis=1
            )
        if self.memory_embeddings.shape != (len(bank), config.input_dim):
            raise ValueError(
                f"NCF memory embedding shape mismatch: {self.memory_embeddings.shape}"
            )

    @torch.no_grad()
    def _rerank_projected(
        self,
        task: np.ndarray,
        baseline_hits: list[dict[str, float | int]],
        *,
        top_k: int,
        pair_features: np.ndarray | None = None,
    ) -> list[dict[str, float | int]]:
        if not baseline_hits:
            return []
        if task.shape != (self.memory_embeddings.shape[1],):
            raise ValueError(f"NCF task embedding shape mismatch: {task.shape}")
        memory_indices = np.asarray(
            [int(hit["memory_index"]) for hit in baseline_hits], dtype=np.int64
        )
        task_batch = torch.from_numpy(np.repeat(task[None, :], len(memory_indices), axis=0))
        memory_batch = torch.from_numpy(self.memory_embeddings[memory_indices])
        if self.pair_feature_names:
            if pair_features is None or pair_features.shape != (
                len(memory_indices), len(self.pair_feature_names)
            ):
                raise ValueError("missing or malformed pair structured features")
            logits = self.model(
                task_batch, memory_batch, torch.from_numpy(pair_features.astype(np.float32))
            )
        else:
            logits = self.model(task_batch, memory_batch)
        if self.ranking_mode == "direct_neumf":
            direct_scores = logits.cpu().numpy()
            reranked = []
            for hit, score in zip(baseline_hits, direct_scores, strict=True):
                row = dict(hit)
                row["query_cosine"] = float(hit["cosine"])
                row["ncf_score"] = float(score)
                row["final_score"] = float(score)
                reranked.append(row)
            reranked.sort(
                key=lambda row: (-float(row["final_score"]), int(row["memory_index"]))
            )
            return reranked[:top_k]

        corrections = (self.correction_scale * torch.tanh(logits)).cpu().numpy()

        reranked = []
        for hit, correction in zip(baseline_hits, corrections, strict=True):
            row = dict(hit)
            row["query_cosine"] = float(hit["cosine"])
            row["ncf_correction"] = float(correction)
            row["final_score"] = float(hit["cosine"]) + float(correction)
            reranked.append(row)
        anchor_count = min(self.preserve_query_cosine_top, len(reranked), top_k)
        anchors = reranked[:anchor_count]
        remainder = reranked[anchor_count:]
        remainder.sort(
            key=lambda row: (-float(row["final_score"]), int(row["memory_index"]))
        )
        return (anchors + remainder)[:top_k]

    def rerank(
        self,
        task_embedding: Iterable[float],
        baseline_hits: list[dict[str, float | int]],
        *,
        top_k: int,
    ) -> list[dict[str, float | int]]:
        if self.structured_feature_config is not None or self.pair_feature_names:
            raise ValueError("structured NCF checkpoint requires rerank_query()")
        task = np.asarray(task_embedding, dtype=np.float32)
        if self.input_projection is not None:
            mean = self.input_projection["mean"].cpu().numpy()
            components = self.input_projection["components"].cpu().numpy()
            task = (task - mean) @ components
        return self._rerank_projected(task, baseline_hits, top_k=top_k)

    def rerank_query(
        self,
        task_query: str,
        task_embedding: Iterable[float],
        baseline_hits: list[dict[str, float | int]],
        *,
        top_k: int,
    ) -> list[dict[str, float | int]]:
        task = np.asarray(task_embedding, dtype=np.float32)
        if self.input_projection is not None:
            mean = self.input_projection["mean"].cpu().numpy()
            components = self.input_projection["components"].cpu().numpy()
            task = (task - mean) @ components
        if self.structured_feature_config is not None:
            task = np.concatenate(
                (task, structured_features(task_query, self.structured_feature_config))
            )
        pair_features = None
        if self.pair_feature_names:
            task_signature = parse_query(task_query).as_dict()
            pair_features = np.stack(
                [
                    relation_features(
                        task_signature,
                        self.memory_signatures[int(hit["memory_index"])],
                        self.memory_validity[int(hit["memory_index"])],
                    )
                    for hit in baseline_hits
                ]
            )
        return self._rerank_projected(
            task, baseline_hits, top_k=top_k, pair_features=pair_features
        )
