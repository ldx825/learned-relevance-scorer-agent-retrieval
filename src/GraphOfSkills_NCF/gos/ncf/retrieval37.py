"""Offline/online retrieval over the shared 37-skill ALFWorld library.

The baseline uses frozen text embeddings as seeds and propagates them through
the frozen SkillDAG graph with personalized PageRank.  The hybrid keeps that
candidate generator and uses the trained content-based NeuMF model only as a
reranker.  Both methods therefore see exactly the same skills and graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gos.core.retrieval import (
    build_personalization,
    personalized_pagerank,
)
from gos.ncf.models import NCFConfig, NeuMF


EDGE_WEIGHTS = {
    "depends_on": 1.0,
    "composes_with": 0.7,
    "specializes": 0.4,
    "similar_to": 0.3,
}

REVERSE_WEIGHTS = {
    "depends_on": 0.2,
    "composes_with": 0.7,
    "specializes": 0.2,
    "similar_to": 1.0,
}


@dataclass(frozen=True)
class RankedSkill:
    skill_id: str
    score: float
    graph_score: float
    cosine_score: float
    ncf_score: float | None = None


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class ALFWorld37Retriever:
    """Shared candidate generator with optional NeuMF reranking."""

    def __init__(
        self,
        *,
        graph_path: Path | str,
        embeddings_path: Path | str,
        model_path: Path | str | None = None,
        seed_top_k: int = 5,
        candidate_top_k: int = 12,
        damping: float = 0.2,
    ) -> None:
        graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
        embedding_rows: dict[str, dict[str, Any]] = json.loads(
            Path(embeddings_path).read_text(encoding="utf-8")
        )
        self.skill_ids = sorted(set(graph["nodes"]) & set(embedding_rows))
        if len(self.skill_ids) != 37:
            raise ValueError(f"expected the shared 37 skills, found {len(self.skill_ids)}")

        self.index = {skill_id: i for i, skill_id in enumerate(self.skill_ids)}
        self.skill_embeddings = np.vstack(
            [
                np.asarray(embedding_rows[skill_id]["embedding"], dtype=np.float32)
                for skill_id in self.skill_ids
            ]
        )
        # NeuMF was trained on the provider's raw embedding vectors.  Keep those
        # vectors intact for model inference and maintain a separate normalized
        # matrix only for cosine candidate generation.
        self.skill_embeddings_normalized = np.vstack(
            [_unit(row) for row in self.skill_embeddings]
        )
        self.transition = self._transition(graph["edges"])
        self.seed_top_k = seed_top_k
        self.candidate_top_k = candidate_top_k
        self.damping = damping

        self.model: NeuMF | None = None
        if model_path is not None:
            config = NCFConfig(input_dim=int(self.skill_embeddings.shape[1]))
            model = NeuMF(config)
            state = torch.load(
                Path(model_path), map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)
            model.eval()
            self.model = model

    def _transition(self, edges: list[dict[str, Any]]) -> np.ndarray:
        count = len(self.skill_ids)
        matrix = np.zeros((count, count), dtype=np.float64)
        for edge in edges:
            source = self.index.get(str(edge.get("source", "")))
            target = self.index.get(str(edge.get("target", "")))
            edge_type = str(edge.get("type", ""))
            if source is None or target is None or edge_type not in EDGE_WEIGHTS:
                continue
            forward = EDGE_WEIGHTS[edge_type]
            matrix[source, target] += forward
            matrix[target, source] += forward * REVERSE_WEIGHTS[edge_type]

        row_sums = matrix.sum(axis=1)
        for i in range(count):
            if row_sums[i] > 0:
                matrix[i] /= row_sums[i]
            else:
                matrix[i, i] = 1.0
        return matrix

    def graph_rank(self, task_embedding: np.ndarray) -> list[RankedSkill]:
        query = _unit(np.asarray(task_embedding, dtype=np.float32))
        cosine = self.skill_embeddings_normalized @ query
        seeds = np.argsort(-cosine, kind="stable")[: self.seed_top_k]
        seed_weights = np.maximum(cosine[seeds], 0.0)
        if float(seed_weights.sum()) <= 0:
            seed_weights = None
        personalization = build_personalization(
            len(self.skill_ids),
            seeds.tolist(),
            None if seed_weights is None else seed_weights.tolist(),
        )
        graph_scores = personalized_pagerank(
            self.transition,
            personalization,
            damping=self.damping,
        )
        order = sorted(
            range(len(self.skill_ids)),
            key=lambda i: (-float(graph_scores[i]), self.skill_ids[i]),
        )
        return [
            RankedSkill(
                skill_id=self.skill_ids[i],
                score=float(graph_scores[i]),
                graph_score=float(graph_scores[i]),
                cosine_score=float(cosine[i]),
            )
            for i in order
        ]

    @torch.no_grad()
    def hybrid_rank(self, task_embedding: np.ndarray) -> list[RankedSkill]:
        if self.model is None:
            raise RuntimeError("hybrid retrieval requires model_path")
        graph_rows = self.graph_rank(task_embedding)[: self.candidate_top_k]
        candidate_indices = [self.index[row.skill_id] for row in graph_rows]
        task = torch.from_numpy(
            np.repeat(
                np.asarray(task_embedding, dtype=np.float32)[None, :],
                len(candidate_indices),
                axis=0,
            )
        )
        skill = torch.from_numpy(self.skill_embeddings[candidate_indices].copy())
        scores = self.model(task, skill).cpu().numpy()
        rows = [
            RankedSkill(
                skill_id=graph_row.skill_id,
                score=float(ncf_score),
                graph_score=graph_row.graph_score,
                cosine_score=graph_row.cosine_score,
                ncf_score=float(ncf_score),
            )
            for graph_row, ncf_score in zip(graph_rows, scores)
        ]
        return sorted(rows, key=lambda row: (-row.score, row.skill_id))

    @torch.no_grad()
    def score_subset(
        self, task_embedding: np.ndarray, skill_ids: list[str]
    ) -> dict[str, float]:
        """Return NeuMF logits for a named subset without changing candidates."""
        if self.model is None:
            raise RuntimeError("score_subset requires model_path")
        missing = sorted(set(skill_ids) - set(self.index))
        if missing:
            raise ValueError(
                "NeuMF checkpoint only covers the shared 37-skill library; "
                f"unknown candidates: {missing}"
            )
        if not skill_ids:
            return {}

        indices = [self.index[skill_id] for skill_id in skill_ids]
        task = torch.from_numpy(
            np.repeat(
                np.asarray(task_embedding, dtype=np.float32)[None, :],
                len(indices),
                axis=0,
            )
        )
        skill = torch.from_numpy(self.skill_embeddings[indices].copy())
        logits = self.model(task, skill).cpu().numpy()
        return {
            skill_id: float(score)
            for skill_id, score in zip(skill_ids, logits)
        }

    @torch.no_grad()
    def score_embedding_subset(
        self,
        task_embedding: np.ndarray,
        skill_embeddings: dict[str, list[float] | np.ndarray],
    ) -> dict[str, float]:
        """Score arbitrary content embeddings, including unseen skill IDs."""
        if self.model is None:
            raise RuntimeError("score_embedding_subset requires model_path")
        if not skill_embeddings:
            return {}
        skill_ids = list(skill_embeddings)
        matrix = np.asarray(
            [skill_embeddings[skill_id] for skill_id in skill_ids],
            dtype=np.float32,
        )
        task = torch.from_numpy(
            np.repeat(
                np.asarray(task_embedding, dtype=np.float32)[None, :],
                len(skill_ids),
                axis=0,
            )
        )
        logits = self.model(task, torch.from_numpy(matrix)).cpu().numpy()
        return {
            skill_id: float(score)
            for skill_id, score in zip(skill_ids, logits)
        }

    def retrieve(
        self, task_embedding: np.ndarray, *, method: str, top_k: int = 3
    ) -> list[RankedSkill]:
        if method == "graph":
            rows = self.graph_rank(task_embedding)
        elif method == "graph_ncf":
            rows = self.hybrid_rank(task_embedding)
        else:
            raise ValueError("method must be 'graph' or 'graph_ncf'")
        return rows[:top_k]
