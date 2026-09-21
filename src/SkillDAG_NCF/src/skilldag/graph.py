"""SkillGraph backend: single global JSON graph + agent-first JSON operations.

Mutation model (2026-04-25):

- ``propose_edge`` / ``propose_remove_edge`` / ``propose_retype_edge`` are PURE
  DRY-RUN. They never modify the graph. Each returns ``{"would": ...,
  "related": [...]}`` where ``related`` lists every edge and history entry
  whose endpoints match the target pair (ignoring direction), so an agent
  can read prior reasons before committing.
- ``edit_edge(action, ...)`` is the ONLY path that writes. Actions:
  ``add`` / ``remove`` / ``retype``. Every edit is immediate, records a
  single-task history entry, and bumps ``updated_at`` via ``save()``.

The evidence-accumulator / 3-task threshold mechanism has been removed: every
mutation is an agent-driven, evidence-in-hand decision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .search import DEFAULT_SKILLS_DIR

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTAINER_SKILLDAG_GRAPH_PATH = Path("/var/lib/skilldag/runtime/skillgraph.json")
_DEFAULT_SKILLDAG_GRAPH_PATH = _REPO_ROOT / "data" / "skilldag_graphs" / "skillgraph_200.json"
DEFAULT_GRAPH_PATH = (
    _CONTAINER_SKILLDAG_GRAPH_PATH
    if _CONTAINER_SKILLDAG_GRAPH_PATH.exists()
    else _DEFAULT_SKILLDAG_GRAPH_PATH
)
INITIALIZE_ENABLED_ENV = "SKILLDAG_INITIALIZE_ENABLED"

EDGE_TYPES = {
    "depends_on",
    "composes_with",
    "similar_to",
    "conflicts_with",
    "specializes",
}

SYMMETRIC_EDGE_TYPES = {
    "composes_with",
    "similar_to",
    "conflicts_with",
}

# Edge types the neighbor BFS is allowed to traverse. `conflicts_with` is
# intentionally excluded: it is a "do not co-select" signal, not a navigable
# relation, so walking across it would surface pairs the agent is supposed to
# keep apart.
WALKABLE_EDGE_TYPES: set[str] = {
    "specializes",
    "composes_with",
    "depends_on",
    "similar_to",
}
DEFAULT_SEARCH_DEPTH = 2

HISTORY_ACTIONS = {"add_edge", "remove_edge", "retype_edge", "rollback_edge"}
EDIT_ACTIONS = {"add", "remove", "retype"}
# The directed acyclic backbone the system name refers to. Symmetric
# relations are layered on top and are exempt from the cycle check.
BACKBONE_EDGE_TYPES = ("depends_on", "specializes")

DEFAULT_SCHEMA_VERSION = "skillgraph.v1"
MAX_REASON_CHARS = 200
RELATED_HISTORY_CAP = 5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class GraphCLIError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }


def _parse_skill_frontmatter(skill_md: Path) -> dict[str, Any]:
    if not skill_md.exists():
        return {}
    text = skill_md.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end is None:
        return {}
    data: dict[str, Any] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"name", "description"}:
            data[key] = value
    return data


@dataclass(frozen=True)
class NormalizedEdge:
    source: str
    target: str
    edge_type: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type,
        }

    def key(self) -> str:
        return f"{self.edge_type}:{self.source}:{self.target}"


class SkillGraph:
    def __init__(self, graph_path: Path, skills_dir: Path, data: dict[str, Any]):
        self.graph_path = graph_path
        self.skills_dir = skills_dir
        self.data = data

    @classmethod
    def load(
        cls,
        graph_path: str | Path = DEFAULT_GRAPH_PATH,
        skills_dir: str | Path = DEFAULT_SKILLS_DIR,
    ) -> "SkillGraph":
        graph_path = Path(graph_path)
        skills_dir = Path(skills_dir)

        if graph_path.exists():
            raw = json.loads(graph_path.read_text(encoding="utf-8"))
        elif os.environ.get(INITIALIZE_ENABLED_ENV) == "1":
            logger.info("skillgraph.json missing; SKILLDAG_INITIALIZE_ENABLED=1 → running cold-start initialization")
            raw = cls._initialize_graph(skills_dir, build_edges=True)
        else:
            raise GraphCLIError(
                "GRAPH_NOT_FOUND",
                f"skillgraph.json not found at {graph_path}. "
                f"To auto-build a cold-start graph (build-time only), set "
                f"SKILLDAG_INITIALIZE_ENABLED=1 and ensure SKILLDAG_EMBEDDING_API_KEY "
                f"and SKILLDAG_LLM_API_KEY are set.",
            )

        # Strip any legacy proposals container; the accumulator is gone.
        changed = "proposals" in raw
        raw.pop("proposals", None)

        graph = cls(graph_path=graph_path, skills_dir=skills_dir, data=raw)
        changed = graph._sync_nodes_from_skills_dir() or changed
        graph.validate()
        if changed:
            graph.save()
        return graph

    @staticmethod
    def _initialize_graph(skills_dir: Path, build_edges: bool = False) -> dict[str, Any]:
        """Build initial graph structure.

        Args:
            skills_dir: path to skills directory to discover nodes from
            build_edges: if True, call cold-start initialization to generate similar_to /
                         specializes edges via embedding + LLM.
                         If False, leave edges empty (legacy behavior).
        """
        nodes = SkillGraph._discover_nodes(skills_dir)
        edges: list[dict[str, Any]] = []
        if build_edges and nodes:
            from .initialize import initialize_edges, InitializationError
            try:
                edges = initialize_edges(nodes)
            except InitializationError as exc:
                raise GraphCLIError("INITIALIZE_FAILED", str(exc)) from exc
            except Exception as exc:  # network / API errors
                raise GraphCLIError("INITIALIZE_FAILED", f"cold-start initialization failed: {exc}") from exc
        return {
            "schema_version": DEFAULT_SCHEMA_VERSION,
            "updated_at": utc_now_iso(),
            "nodes": nodes,
            "edges": edges,
            "history": [],
        }

    @staticmethod
    def _discover_nodes(skills_dir: Path) -> dict[str, Any]:
        nodes: dict[str, Any] = {}
        if not skills_dir.exists():
            return nodes
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                continue
            metadata = _parse_skill_frontmatter(skill_md)
            skill_id = entry.name
            nodes[skill_id] = {
                "name": metadata.get("name", skill_id),
                "description": metadata.get("description", ""),
                "path": str(entry),
                "status": "active",
                "tags": [],
            }
        return nodes

    def _sync_nodes_from_skills_dir(self) -> bool:
        discovered = self._discover_nodes(self.skills_dir)
        nodes = self.data.setdefault("nodes", {})
        changed = False
        for skill_id, node in discovered.items():
            existing = nodes.get(skill_id)
            if existing is None:
                nodes[skill_id] = node
                changed = True
                continue
            if node.get("path") and existing.get("path") != node.get("path"):
                existing["path"] = node["path"]
                changed = True
            for field in ("path", "name", "description"):
                if not existing.get(field) and node.get(field):
                    existing[field] = node[field]
                    changed = True
            if "status" not in existing:
                existing["status"] = "active"
                changed = True
            if "tags" not in existing:
                existing["tags"] = []
                changed = True
        if changed:
            self.data["updated_at"] = utc_now_iso()
        return changed

    def save(self) -> None:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = utc_now_iso()
        # Proposals field is obsolete; do not re-emit it on save.
        self.data.pop("proposals", None)
        tmp_path = self.graph_path.with_name(f".{self.graph_path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, self.graph_path)

    def validate(self) -> None:
        if "history" not in self.data:
            self.data["history"] = []
        if self.data.get("schema_version") != DEFAULT_SCHEMA_VERSION:
            raise GraphCLIError("INVALID_SCHEMA_VERSION", "Unsupported or missing skillgraph schema_version")

        if not isinstance(self.data.get("nodes"), dict):
            raise GraphCLIError("INVALID_GRAPH", "`nodes` must be an object")
        if not isinstance(self.data.get("edges"), list):
            raise GraphCLIError("INVALID_GRAPH", "`edges` must be a list")
        if not isinstance(self.data.get("history"), list):
            raise GraphCLIError("INVALID_GRAPH", "`history` must be a list")

        nodes = self.data["nodes"]
        seen_edges: set[str] = set()
        for edge in self.data["edges"]:
            normalized = self._normalize_edge(
                edge.get("source", ""),
                edge.get("target", ""),
                edge.get("type", ""),
            )
            if normalized.source not in nodes or normalized.target not in nodes:
                raise GraphCLIError(
                    "EDGE_NODE_NOT_FOUND",
                    f"Edge references missing node: {normalized.source} -> {normalized.target}",
                )
            if normalized.source == normalized.target:
                raise GraphCLIError("SELF_LOOP_NOT_ALLOWED", f"Self-loop not allowed for {normalized.source}")
            key = normalized.key()
            if key in seen_edges:
                raise GraphCLIError("DUPLICATE_EDGE", f"Duplicate edge detected: {key}")
            seen_edges.add(key)
            edge["source"] = normalized.source
            edge["target"] = normalized.target
            edge["type"] = normalized.edge_type

        for entry in self.data["history"]:
            action = entry.get("action")
            if action not in HISTORY_ACTIONS:
                raise GraphCLIError("INVALID_HISTORY", f"Invalid history action: {action}")
            if not entry.get("applied_at"):
                raise GraphCLIError("INVALID_HISTORY", "History entry missing applied_at")

    def _normalize_edge(self, source: str, target: str, edge_type: str) -> NormalizedEdge:
        if edge_type not in EDGE_TYPES:
            raise GraphCLIError("INVALID_EDGE_TYPE", f"Unsupported edge type: {edge_type}")
        if not source or not target:
            raise GraphCLIError("INVALID_EDGE", "source and target are required")
        if source == target:
            raise GraphCLIError("SELF_LOOP_NOT_ALLOWED", f"Self-loop not allowed for {source}")
        if edge_type in SYMMETRIC_EDGE_TYPES:
            source, target = sorted((source, target))
        return NormalizedEdge(source=source, target=target, edge_type=edge_type)

    def _edge_lookup(self) -> dict[str, dict[str, Any]]:
        return {
            self._normalize_edge(edge["source"], edge["target"], edge["type"]).key(): edge
            for edge in self.data["edges"]
        }

    # ------------------------------------------------------------------
    # Online-edit guardrails. The propose-then-commit protocol stays
    # agent-driven (single-observation commit), but these checks give it
    # explicit safety properties instead of relying on agent discipline
    # alone: (1) the depends_on/specializes backbone stays acyclic,
    # (2) a skill pair cannot be simultaneously co-selectable and
    # mutually exclusive, and (3) any committed edit is reversible.
    # ------------------------------------------------------------------

    def _backbone_reachable(self, src: str, dst: str) -> bool:
        """True if ``dst`` is reachable from ``src`` along directed
        backbone edges (``depends_on`` / ``specializes``) only."""
        if src == dst:
            return True
        adj: dict[str, list[str]] = {}
        for e in self.data["edges"]:
            if e["type"] in BACKBONE_EDGE_TYPES:
                adj.setdefault(e["source"], []).append(e["target"])
        seen, stack = {src}, [src]
        while stack:
            for nxt in adj.get(stack.pop(), ()):
                if nxt == dst:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def _assert_acyclic(self, source: str, target: str, edge_type: str) -> None:
        if edge_type in BACKBONE_EDGE_TYPES and self._backbone_reachable(target, source):
            raise GraphCLIError(
                "WOULD_CREATE_CYCLE",
                f"{edge_type}({source} -> {target}) would close a cycle in the "
                f"depends_on/specializes backbone ({target} already reaches {source}); "
                f"retract or retype the conflicting edge first.",
            )

    @staticmethod
    def _edge_polarity(edge_type: str) -> str:
        """``conflicts_with`` marks "do not co-select"; every other type
        asserts a positive relation. The two polarities are mutually
        exclusive for one skill pair."""
        return "exclude" if edge_type == "conflicts_with" else "relate"

    def _assert_no_contradiction(
        self,
        source: str,
        target: str,
        edge_type: str,
        *,
        ignore_edge_key: str | None = None,
    ) -> None:
        pair = {source, target}
        new_pol = self._edge_polarity(edge_type)
        for e in self.data["edges"]:
            existing = self._normalize_edge(e["source"], e["target"], e["type"])
            if ignore_edge_key and existing.key() == ignore_edge_key:
                continue
            if {e["source"], e["target"]} == pair and e["type"] != edge_type:
                if self._edge_polarity(e["type"]) != new_pol:
                    raise GraphCLIError(
                        "CONTRADICTORY_EDGE",
                        f"{edge_type}({source}, {target}) contradicts existing "
                        f"{e['type']}({e['source']}, {e['target']}): a pair cannot be "
                        f"both co-selectable and mutually exclusive; remove or retype "
                        f"the existing edge first.",
                    )

    def _ensure_skill_exists(self, skill_id: str) -> None:
        if skill_id not in self.data["nodes"]:
            raise GraphCLIError("SKILL_NOT_FOUND", f"Skill not found: {skill_id}")

    def _find_neighbors(self, skill_id: str, edge_type: str | None = None) -> list[dict[str, Any]]:
        self._ensure_skill_exists(skill_id)
        results = []
        for edge in self.data["edges"]:
            if edge_type and edge["type"] != edge_type:
                continue
            if edge["source"] == skill_id:
                results.append({**edge, "direction": "outgoing"})
            elif edge["target"] == skill_id:
                results.append({**edge, "direction": "incoming"})
        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        depth: int = DEFAULT_SEARCH_DEPTH,
    ) -> dict[str, Any]:
        """Vector top-K matches plus a separate depth-bounded neighbor list.

        Three sections, no score mixing:

        - ``matches``: top ``top_k`` skills ranked by cosine similarity
          between the query embedding and each node's e_self embedding
          (skill_id + description + body preview, same view used by
          cold-start initialization). Replicates the GoS ``vectorskills``
          baseline retrieval. Embeddings are computed via
          ``SKILLDAG_EMBEDDING_MODEL`` (default ``text-embedding-3-large``)
          and cached at ``<graph_path>.embeddings.json``. Each entry's
          ``score`` is the float cosine in [-1, 1]. The graph is NOT
          consulted for this list.

        - ``neighbors``: every unique skill reachable from ANY match within
          ``depth`` hops along walkable edges (``specializes``,
          ``composes_with``, ``depends_on``, ``similar_to``). Each edge is
          traversed in both directions. ``conflicts_with`` is NOT traversed
          (prune-only signal). Each neighbor entry records the shortest
          ``depth`` it was reached at, the immediate predecessor
          (``reached_from``), and the edge type used (``via``). Skills that
          appear in ``matches`` are excluded from ``neighbors`` so each skill
          shows up exactly once.

        - ``conflicts``: direct (1-hop) ``conflicts_with`` edges incident to
          any match. Surfaces "DO NOT co-load with this match" pairs as a
          separate, opposite-polarity list from neighbors. No transitive
          expansion — conflicts-of-conflicts are semantically undefined.
          Cold-start initialization (post 2026-04-24) does not emit
          ``conflicts_with``; this section only becomes non-empty once the
          online pipeline accumulates execution-evidenced conflicts.

        Args:
            query: free-text query.
            top_k: cap on ``matches`` (cosine top-K).
            depth: BFS depth for ``neighbors``. ``0`` returns no neighbors.

        Returns:
            ``{"matches": [...], "neighbors": [...], "conflicts": [...]}``.
        """
        retrieval_query = self._ncf_context_query(query)
        scored, query_embedding, node_embeddings = self._vector_score_context(retrieval_query)
        ncf_scores: dict[str, float] = {}
        plan_safe_ids: set[str] = set()
        cosine_scores = {sid: float(score) for sid, score in scored}
        if os.environ.get("SKILLDAG_NCF_MODEL"):
            scored, ncf_scores, plan_safe_ids = self._ncf_rerank(
                retrieval_query,
                scored,
                top_k=top_k,
                query_embedding=query_embedding,
                node_embeddings=node_embeddings,
            )
        match_ids: list[str] = [sid for sid, _ in scored[:top_k]]
        match_score_by_id: dict[str, float] = {
            sid: float(score) for sid, score in scored[:top_k]
        }
        matches = [
            self._format_match_entry(sid, score=match_score_by_id[sid])
            for sid in match_ids
        ]
        if ncf_scores:
            for entry in matches:
                sid = entry["skill_id"]
                entry["cosine_score"] = cosine_scores[sid]
                entry["ncf_score"] = ncf_scores[sid]
                entry["score_type"] = "calibrated_hybrid"
                entry["plan_safe"] = sid in plan_safe_ids
        neighbors = self._collect_neighbors(match_ids, depth=depth)
        conflicts = self._collect_conflicts(match_ids)
        self._audit_search(
            query=query,
            top_k=top_k,
            returned_count=len(matches),
            ncf_enabled=bool(os.environ.get("SKILLDAG_NCF_MODEL")),
            ncf_reranked=bool(ncf_scores),
        )
        return {"matches": matches, "neighbors": neighbors, "conflicts": conflicts}

    def _audit_search(
        self,
        *,
        query: str,
        top_k: int,
        returned_count: int,
        ncf_enabled: bool,
        ncf_reranked: bool,
    ) -> None:
        """Append one compact runtime record for every completed search query.

        The benchmark mounts the audit path inside each task's writable
        SkillDAG runtime directory.  A single ``os.write`` to an O_APPEND file
        keeps each JSONL record intact even when multiple CLI processes are
        launched close together.  Query text is never persisted; only its
        hash is retained for invocation counting and duplicate analysis.
        """
        configured = os.environ.get("SKILLDAG_NCF_AUDIT_LOG", "").strip()
        if not configured:
            return
        alpha, priors = self._ncf_calibration(
            os.environ.get("SKILLDAG_NCF_MODEL", "").strip()
        )
        payload = {
            "schema_version": "skilldag.ncf_search_audit.v1",
            "timestamp": utc_now_iso(),
            "pid": os.getpid(),
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "top_k": int(top_k),
            "returned_count": int(returned_count),
            "ncf_enabled": bool(ncf_enabled),
            "ncf_reranked": bool(ncf_reranked),
            "candidate_k": int(os.environ.get("SKILLDAG_NCF_CANDIDATE_K", "12")),
            "alpha_cosine": float(alpha),
            "nonzero_prior_count": sum(abs(float(value)) > 0.0 for value in priors.values()),
        }
        path = Path(configured)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)

    def _ncf_rerank(
        self,
        query: str,
        cosine_ranking: list[tuple[str, float]],
        *,
        top_k: int,
        query_embedding: list[float] | None = None,
        node_embeddings: dict[str, list[float]] | None = None,
    ) -> tuple[list[tuple[str, float]], dict[str, float], set[str]]:
        """Conservatively rerank a cosine shortlist with calibrated NeuMF.

        Runtime search uses the same scoring contract as task-start planning:
        remove learned skill-popularity priors, z-normalize cosine and NeuMF
        specificity scores, blend them using the bundle's calibrated alpha,
        and apply graph-aware diversity.  A task-start plan contributes at
        most one safety result selected by the current query's cosine score;
        this prevents a terse online query from silently evicting every
        high-confidence proactive recommendation.
        """
        model_path = os.environ.get("SKILLDAG_NCF_MODEL", "").strip()
        if not model_path:
            return cosine_ranking, {}, set()

        candidate_k = max(
            top_k,
            int(os.environ.get("SKILLDAG_NCF_CANDIDATE_K", "12")),
        )
        candidates = cosine_ranking[:candidate_k]
        if not candidates:
            return [], {}, set()

        cosine_scores = {sid: float(score) for sid, score in cosine_ranking}
        planned_ids = self._ncf_plan_skill_ids()
        candidate_ids_seen = {sid for sid, _ in candidates}
        for skill_id in planned_ids:
            if skill_id in cosine_scores and skill_id not in candidate_ids_seen:
                candidates.append((skill_id, cosine_scores[skill_id]))
                candidate_ids_seen.add(skill_id)

        node_embeddings = node_embeddings or self._load_or_build_node_embeddings()
        expected_pool_size = os.environ.get("SKILLDAG_NCF_EXPECTED_POOL_SIZE", "").strip()
        if expected_pool_size and len(node_embeddings) != int(expected_pool_size):
            raise GraphCLIError(
                "NCF_SKILL_POOL_MISMATCH",
                f"NCF mode expects {expected_pool_size} skills, but the current graph "
                f"contains {len(node_embeddings)} embeddings.",
            )
        query_embedding = query_embedding or self._embed_query(query)

        from .ncf_reranker import load_reranker

        reranker = load_reranker(
            model_path,
            input_dim=len(query_embedding),
        )
        candidate_ids = [sid for sid, _ in candidates]
        if hasattr(reranker, "score_by_id"):
            logits = reranker.score_by_id(query_embedding, candidate_ids)
        else:
            logits = reranker.score(
                query_embedding,
                [node_embeddings[sid] for sid in candidate_ids],
            )
        ncf_scores = dict(zip(candidate_ids, logits))
        alpha, priors = self._ncf_calibration(model_path)
        from .hybrid_selection import normalize_scores, select_diverse

        specific_scores = [
            ncf_scores[sid] - float(priors.get(sid, 0.0))
            for sid in candidate_ids
        ]
        cosine_z = normalize_scores([cosine_scores[sid] for sid in candidate_ids])
        ncf_z = normalize_scores(specific_scores)
        rows = [
            {
                "skill_id": sid,
                "base_score": alpha * cz + (1.0 - alpha) * nz,
            }
            for sid, cz, nz in zip(candidate_ids, cosine_z, ncf_z)
        ]
        rows.sort(key=lambda row: (-float(row["base_score"]), row["skill_id"]))
        cosine_safe_ids = {
            sid
            for sid, _ in sorted(
                candidates, key=lambda item: (-item[1], item[0])
            )[: min(3, len(candidates))]
        }
        selected = select_diverse(
            rows,
            edges=self.data.get("edges", []),
            cosine_safe_ids=cosine_safe_ids,
            top_k=top_k,
        )
        plan_safe_ids: set[str] = set()
        selected_ids = {row["skill_id"] for row in selected}
        if planned_ids and not selected_ids.intersection(planned_ids):
            planned_rows = sorted(
                (row for row in rows if row["skill_id"] in planned_ids),
                key=lambda row: (
                    -cosine_scores[row["skill_id"]],
                    row["skill_id"],
                ),
            )
            if planned_rows:
                safety = planned_rows[0]
                if len(selected) < top_k:
                    selected.append(safety)
                else:
                    replace_at = next(
                        (
                            index
                            for index in range(len(selected) - 1, -1, -1)
                            if selected[index]["skill_id"] not in cosine_safe_ids
                        ),
                        len(selected) - 1,
                    )
                    selected[replace_at] = safety
                plan_safe_ids.add(safety["skill_id"])

        selected_ids = {row["skill_id"] for row in selected}
        plan_safe_ids.update(selected_ids.intersection(planned_ids))
        remaining = [
            row for row in rows if row["skill_id"] not in selected_ids
        ]
        ranked_rows = selected + remaining
        ranked = [
            (row["skill_id"], float(row["base_score"]))
            for row in ranked_rows
        ]
        return ranked, ncf_scores, plan_safe_ids

    def _ncf_plan_skill_ids(self) -> list[str]:
        """Return unique task-start plan IDs without consulting gold data."""
        configured = os.environ.get("SKILLDAG_NCF_PLAN", "").strip()
        path = (
            Path(configured)
            if configured
            else self.graph_path.parent / "ncf_plan.json"
        )
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        output: list[str] = []
        for view in payload.get("views", []):
            for match in view.get("matches", []):
                skill_id = str(match.get("skill_id", "")).strip()
                if skill_id and skill_id not in output:
                    output.append(skill_id)
        return output

    def _ncf_context_query(self, query: str) -> str:
        """Use the V14 full-task/current-phase view when hybrid mode is active."""
        if not os.environ.get("SKILLDAG_NCF_MODEL"):
            return query
        configured = os.environ.get("SKILLDAG_NCF_PLAN", "").strip()
        path = Path(configured) if configured else self.graph_path.parent / "ncf_plan.json"
        if not path.is_file():
            return query
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return query
        full_task = str(payload.get("full_task", "")).strip()
        if not full_task:
            return query
        return f"Full task:\n{full_task}\n\nCurrent phase/query:\n{query}"

    @staticmethod
    def _ncf_calibration(model_path: str) -> tuple[float, dict[str, float]]:
        """Load portable-bundle calibration; checkpoints keep safe defaults."""
        alpha = float(os.environ.get("SKILLDAG_NCF_ALPHA", "0.25"))
        priors: dict[str, float] = {}
        path = Path(model_path)
        if path.suffix.lower() == ".json" and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                calibration = payload.get("calibration") or {}
                alpha = float(calibration.get("alpha", alpha))
                priors = {
                    str(skill_id): float(value)
                    for skill_id, value in (
                        calibration.get("skill_logit_prior") or {}
                    ).items()
                }
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return min(1.0, max(0.0, alpha)), priors

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 10,
        *,
        depth: int = DEFAULT_SEARCH_DEPTH,
    ) -> list[dict[str, Any]]:
        """Run search() over N queries with ONE embedding API call.

        Returns a list aligned with input ``queries``; each element is the same
        ``{matches, neighbors, conflicts}`` dict that ``search()`` returns for
        a single query. Empty / whitespace queries yield empty results.
        """
        clean = [(i, q.strip()) for i, q in enumerate(queries)]
        non_empty = [(i, q) for i, q in clean if q]
        results: list[dict[str, Any]] = [
            {"matches": [], "neighbors": [], "conflicts": []} for _ in queries
        ]
        if not non_empty:
            return results

        node_embeds = self._load_or_build_node_embeddings()
        if not node_embeds:
            return results

        # ONE API call for all non-empty queries.
        from .initialize import _normalize
        retrieval_queries = [self._ncf_context_query(q) for _, q in non_empty]
        q_vecs = self._embed_queries_batch(retrieval_queries)

        # Pre-normalize node embeddings once.
        node_items = list(node_embeds.items())
        node_norms = [(sid, _normalize(vec)) for sid, vec in node_items]

        for (orig_i, _q), retrieval_query, q_vec in zip(
            non_empty, retrieval_queries, q_vecs
        ):
            qn = _normalize(q_vec)
            scored = sorted(
                ((sid, sum(a * b for a, b in zip(qn, vec))) for sid, vec in node_norms),
                key=lambda kv: kv[1],
                reverse=True,
            )
            ncf_scores: dict[str, float] = {}
            plan_safe_ids: set[str] = set()
            cosine_scores = {sid: float(score) for sid, score in scored}
            if os.environ.get("SKILLDAG_NCF_MODEL"):
                scored, ncf_scores, plan_safe_ids = self._ncf_rerank(
                    retrieval_query,
                    scored,
                    top_k=top_k,
                    query_embedding=q_vec,
                    node_embeddings=node_embeds,
                )
            match_ids = [sid for sid, _ in scored[:top_k]]
            match_score_by_id = {sid: float(s) for sid, s in scored[:top_k]}
            matches = [
                self._format_match_entry(sid, score=match_score_by_id[sid])
                for sid in match_ids
            ]
            if ncf_scores:
                for entry in matches:
                    sid = entry["skill_id"]
                    entry["cosine_score"] = cosine_scores[sid]
                    entry["ncf_score"] = ncf_scores[sid]
                    entry["score_type"] = "calibrated_hybrid"
                    entry["plan_safe"] = sid in plan_safe_ids
            neighbors = self._collect_neighbors(match_ids, depth=depth)
            conflicts = self._collect_conflicts(match_ids)
            self._audit_search(
                query=_q,
                top_k=top_k,
                returned_count=len(matches),
                ncf_enabled=bool(os.environ.get("SKILLDAG_NCF_MODEL")),
                ncf_reranked=bool(ncf_scores),
            )
            results[orig_i] = {"matches": matches, "neighbors": neighbors, "conflicts": conflicts}

        return results

    def _embeddings_cache_path(self) -> Path:
        return self.graph_path.with_suffix(".embeddings.json")

    def _node_embed_text(self, skill_id: str, node: dict[str, Any]) -> str:
        """Compose the e_self text for a node (matches initialize.py)."""
        from .initialize import _read_body_preview
        desc = str(node.get("description", ""))
        path = node.get("path", "")
        body = _read_body_preview(str(path)) if path else ""
        return f"{skill_id}: {desc}\n\n{body}".strip()

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _load_or_build_node_embeddings(self) -> dict[str, list[float]]:
        """Return ``{skill_id: embedding}`` for every current node.

        Cache file ``<graph_path>.embeddings.json`` keyed by skill_id stores
        ``{text_hash, model, embedding}``. Entries whose text or model no
        longer matches are recomputed; missing nodes are added; nodes that
        no longer exist are dropped. All embedding work is delegated to
        ``initialize._embed_batch`` so the model / endpoint / retry behaviour
        stays in lock-step with cold-start.
        """
        from .initialize import _embed_batch, InitializationError, DEFAULT_EMBEDDING_MODEL

        cache_path = self._embeddings_cache_path()
        cache: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(cache, dict):
                    cache = {}
            except Exception:
                cache = {}

        model = os.environ.get("SKILLDAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        fresh: dict[str, list[float]] = {}
        to_embed: list[tuple[str, str, str]] = []  # (skill_id, text, hash)

        for skill_id in sorted(self.data["nodes"].keys()):
            node = self.data["nodes"][skill_id]
            text = self._node_embed_text(skill_id, node)
            h = self._text_hash(text)
            entry = cache.get(skill_id)
            if (
                isinstance(entry, dict)
                and entry.get("text_hash") == h
                and entry.get("model") == model
                and isinstance(entry.get("embedding"), list)
            ):
                fresh[skill_id] = entry["embedding"]
            else:
                to_embed.append((skill_id, text, h))

        cache_dirty = bool(to_embed)
        if to_embed:
            texts = [t for _, t, _ in to_embed]
            try:
                embeddings = _embed_batch(texts)
            except InitializationError as exc:
                raise GraphCLIError("EMBEDDING_FAILED", str(exc)) from exc
            except Exception as exc:
                raise GraphCLIError("EMBEDDING_FAILED", f"embedding API failed: {exc}") from exc

            for (sid, _, h), emb in zip(to_embed, embeddings):
                fresh[sid] = emb
                cache[sid] = {"text_hash": h, "model": model, "embedding": emb}

        # Drop cache entries for nodes that no longer exist in the graph.
        existing_ids = set(self.data["nodes"].keys())
        stale_ids = [k for k in cache if k not in existing_ids]
        if stale_ids:
            for k in stale_ids:
                cache.pop(k, None)
            cache_dirty = True

        if cache_dirty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        return fresh

    def _embed_query(self, query: str) -> list[float]:
        from .initialize import _embed_batch, InitializationError
        try:
            return _embed_batch([query])[0]
        except InitializationError as exc:
            raise GraphCLIError("EMBEDDING_FAILED", str(exc)) from exc
        except Exception as exc:
            raise GraphCLIError("EMBEDDING_FAILED", f"embedding API failed: {exc}") from exc

    def _embed_queries_batch(self, queries: list[str]) -> list[list[float]]:
        """Embed N queries in one API call."""
        from .initialize import _embed_batch, InitializationError
        try:
            return _embed_batch(queries)
        except InitializationError as exc:
            raise GraphCLIError("EMBEDDING_FAILED", str(exc)) from exc
        except Exception as exc:
            raise GraphCLIError("EMBEDDING_FAILED", f"embedding API failed: {exc}") from exc

    def _vector_score(self, query: str) -> list[tuple[str, float]]:
        """Cosine similarity vs every node's e_self embedding."""
        scored, _, _ = self._vector_score_context(query)
        return scored

    def _vector_score_context(
        self, query: str
    ) -> tuple[
        list[tuple[str, float]],
        list[float] | None,
        dict[str, list[float]],
    ]:
        """Cosine ranking plus raw vectors, so NCF adds no embedding API call."""
        if not query or not query.strip():
            return [], None, {}
        node_embeds = self._load_or_build_node_embeddings()
        if not node_embeds:
            return [], None, {}

        from .initialize import _normalize
        q_vec = self._embed_query(query)
        qn = _normalize(q_vec)
        scored: list[tuple[str, float]] = []
        for sid, emb in node_embeds.items():
            en = _normalize(emb)
            cos = sum(a * b for a, b in zip(qn, en))
            scored.append((sid, float(cos)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored, q_vec, node_embeds

    def _walkable_neighbors(self, skill_id: str) -> list[tuple[str, str]]:
        """Return ``[(neighbor_id, edge_type)]`` reachable from ``skill_id``
        via walkable edges, both directions, excluding ``conflicts_with``."""
        nodes = self.data["nodes"]
        out: list[tuple[str, str]] = []
        for edge in self.data.get("edges", []):
            etype = edge.get("type", "")
            if etype not in WALKABLE_EDGE_TYPES:
                continue
            src = edge.get("source")
            dst = edge.get("target")
            if src not in nodes or dst not in nodes:
                continue
            if src == skill_id:
                out.append((dst, etype))
            elif dst == skill_id:
                out.append((src, etype))
        return out

    def _collect_neighbors(
        self, match_ids: list[str], depth: int
    ) -> list[dict[str, Any]]:
        if depth <= 0 or not match_ids:
            return []

        matches_set = set(match_ids)
        reached: dict[str, tuple[int, str, str]] = {}
        frontier: list[str] = list(match_ids)

        for hop in range(1, depth + 1):
            next_frontier: list[str] = []
            for src in frontier:
                for neighbor, etype in self._walkable_neighbors(src):
                    if neighbor in matches_set or neighbor in reached:
                        continue
                    reached[neighbor] = (hop, src, etype)
                    next_frontier.append(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier

        results: list[dict[str, Any]] = []
        for sid, (sid_depth, predecessor, via) in reached.items():
            node = self.data["nodes"].get(sid, {})
            results.append(
                {
                    "skill_id": sid,
                    "name": node.get("name", sid),
                    "description": node.get("description", ""),
                    "depth": sid_depth,
                    "reached_from": predecessor,
                    "via": via,
                }
            )
        results.sort(key=lambda r: (r["depth"], r["skill_id"]))
        return results

    def _collect_conflicts(self, match_ids: list[str]) -> list[dict[str, Any]]:
        if not match_ids:
            return []
        nodes = self.data["nodes"]
        matches_set = set(match_ids)
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, Any]] = []
        for edge in self.data.get("edges", []):
            if edge.get("type") != "conflicts_with":
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            if src not in nodes or tgt not in nodes:
                continue
            pairs: list[tuple[str, str]] = []
            if src in matches_set:
                pairs.append((src, tgt))
            if tgt in matches_set:
                pairs.append((tgt, src))
            for match_id, other_id in pairs:
                key = (other_id, match_id)
                if key in seen:
                    continue
                seen.add(key)
                node = nodes.get(other_id, {})
                out.append(
                    {
                        "skill_id": other_id,
                        "name": node.get("name", other_id),
                        "description": node.get("description", ""),
                        "conflicts_with": match_id,
                        "reason": edge.get("reason", ""),
                        "origin": edge.get("origin", ""),
                    }
                )
        out.sort(key=lambda r: (r["conflicts_with"], r["skill_id"]))
        return out

    def _format_match_entry(self, skill_id: str, *, score: float) -> dict[str, Any]:
        node = self.data["nodes"].get(skill_id, {})
        return {
            "skill_id": skill_id,
            "name": node.get("name", skill_id),
            "description": node.get("description", ""),
            "score": float(score),
        }

    def get_skill(self, skill_id: str) -> dict[str, Any]:
        self._ensure_skill_exists(skill_id)
        node = self.data["nodes"][skill_id]
        neighbors = self._find_neighbors(skill_id)
        return {
            "skill_id": skill_id,
            "node": {
                "name": node.get("name", skill_id),
                "description": node.get("description", ""),
                "status": node.get("status", "active"),
                "tags": node.get("tags", []),
            },
            "neighbor_count": len(neighbors),
        }

    def get_dependencies(self, skill_id: str, transitive: bool = False) -> dict[str, Any]:
        self._ensure_skill_exists(skill_id)
        plan_edge_types = {"depends_on", "composes_with"}
        if not transitive:
            deps = [
                {
                    "target": edge["target"],
                    "type": edge["type"],
                }
                for edge in self.data["edges"]
                if edge["source"] == skill_id and edge["type"] in plan_edge_types
            ]
            return {"skill_id": skill_id, "dependencies": deps}

        ordered: list[tuple[str, str]] = []  # (skill_id, edge_type)
        seen: set[str] = set()
        stack = [skill_id]
        while stack:
            current = stack.pop()
            for edge in self.data["edges"]:
                if edge["source"] == current and edge["type"] in plan_edge_types:
                    dep = edge["target"]
                    if dep in seen:
                        continue
                    seen.add(dep)
                    ordered.append((dep, edge["type"]))
                    stack.append(dep)
        return {
            "skill_id": skill_id,
            "dependencies": [{"target": dep, "type": etype} for dep, etype in ordered],
        }

    def get_conflicts(self, skill_id: str) -> dict[str, Any]:
        conflicts = []
        for edge in self.data["edges"]:
            if edge["type"] != "conflicts_with":
                continue
            if edge["source"] == skill_id:
                conflicts.append({"target": edge["target"], "type": edge["type"]})
            elif edge["target"] == skill_id:
                conflicts.append({"target": edge["source"], "type": edge["type"]})
        return {"skill_id": skill_id, "conflicts": conflicts}

    def get_alternatives(self, skill_id: str) -> dict[str, Any]:
        self._ensure_skill_exists(skill_id)
        alternatives = []
        for edge in self.data["edges"]:
            if edge["type"] == "similar_to":
                if edge["source"] == skill_id:
                    alternatives.append({"target": edge["target"], "via": "similar_to"})
                elif edge["target"] == skill_id:
                    alternatives.append({"target": edge["source"], "via": "similar_to"})
            elif edge["type"] == "specializes":
                if edge["source"] == skill_id:
                    alternatives.append({"target": edge["target"], "via": "specializes_parent"})
                elif edge["target"] == skill_id:
                    alternatives.append({"target": edge["source"], "via": "specializes_child"})
        return {"skill_id": skill_id, "alternatives": alternatives}

    def expand_set(self, skill_ids: list[str]) -> dict[str, Any]:
        missing: list[str] = []
        present = list(dict.fromkeys(skill_ids))
        seen = set(present)
        queue = list(present)
        while queue:
            current = queue.pop(0)
            self._ensure_skill_exists(current)
            for edge in self.data["edges"]:
                if edge["source"] == current and edge["type"] == "depends_on":
                    dep = edge["target"]
                    if dep not in seen:
                        seen.add(dep)
                        queue.append(dep)
                        missing.append(dep)
        return {
            "input": present,
            "expanded": present + missing,
            "added": missing,
        }

    def check_set(self, skill_ids: list[str]) -> dict[str, Any]:
        original = list(dict.fromkeys(skill_ids))
        expanded = self.expand_set(original)
        selected = set(original)
        missing_deps = [dep for dep in expanded["added"] if dep not in selected]

        conflicts = []
        for edge in self.data["edges"]:
            if edge["type"] != "conflicts_with":
                continue
            if edge["source"] in selected and edge["target"] in selected:
                conflicts.append({"source": edge["source"], "target": edge["target"], "type": edge["type"]})

        redundant = []
        for edge in self.data["edges"]:
            if edge["source"] not in selected or edge["target"] not in selected:
                continue
            if edge["type"] == "similar_to":
                redundant.append({"pair": [edge["source"], edge["target"]], "reason": "similar_to"})
            elif edge["type"] == "specializes":
                redundant.append(
                    {
                        "pair": [edge["source"], edge["target"]],
                        "reason": "specializes",
                    }
                )

        return {
            "input": original,
            "missing_deps": sorted(dict.fromkeys(missing_deps)),
            "conflicts": conflicts,
            "redundant": redundant,
        }

    def repair_set(self, skill_ids: list[str]) -> dict[str, Any]:
        report = self.check_set(skill_ids)
        remove = []
        for item in report["conflicts"]:
            remove.append(item["target"])
        for item in report["redundant"]:
            pair = item["pair"]
            remove.append(pair[1])
        return {
            "input": report["input"],
            "suggestions": {
                "add": report["missing_deps"],
                "remove": sorted(dict.fromkeys(remove)),
            },
            "check": report,
        }

    # ------------------------------------------------------------------
    # Propose (dry-run) — never mutates. Returns what WOULD happen plus
    # every existing edge / history entry touching the target skill pair
    # so the agent can read prior reasons before committing.
    # ------------------------------------------------------------------

    def _gather_related(self, src: str, tgt: str) -> list[dict[str, Any]]:
        pair = frozenset((src, tgt))
        edge_hits: list[dict[str, Any]] = []
        for edge in self.data.get("edges", []):
            es = edge.get("source", "")
            et = edge.get("target", "")
            if frozenset((es, et)) != pair:
                continue
            edge_hits.append(
                {
                    "kind": "edge",
                    "source": es,
                    "target": et,
                    "type": edge.get("type", ""),
                    "reason": edge.get("reason", ""),
                    "origin": edge.get("origin", ""),
                }
            )

        history_hits: list[dict[str, Any]] = []
        for entry in self.data.get("history", []):
            target_edge = None
            for candidate_key in ("edge", "from_edge", "to_edge"):
                candidate = entry.get(candidate_key) or {}
                if not candidate:
                    continue
                cs = candidate.get("source", "")
                ct = candidate.get("target", "")
                if frozenset((cs, ct)) == pair:
                    target_edge = candidate
                    break
            if target_edge is None:
                continue
            evidence = entry.get("evidence") or []
            task_id = evidence[0].get("task_id") if evidence else ""
            history_hits.append(
                {
                    "kind": "history",
                    "action": entry.get("action", ""),
                    "applied_at": entry.get("applied_at", ""),
                    "reason": entry.get("reason", ""),
                    "task_id": task_id,
                    "type": target_edge.get("type", ""),
                    "source": target_edge.get("source", ""),
                    "target": target_edge.get("target", ""),
                }
            )

        # History is append-only in natural order; keep the most recent entries.
        history_hits = history_hits[-RELATED_HISTORY_CAP:]
        return edge_hits + history_hits

    @staticmethod
    def _validate_reason(reason: str) -> str:
        reason = (reason or "").strip()
        if not reason:
            raise GraphCLIError("INVALID_REASON", "reason is required and must be non-empty")
        return reason[:MAX_REASON_CHARS]

    def propose_edge(
        self, source: str, target: str, edge_type: str, reason: str
    ) -> dict[str, Any]:
        """Dry-run add. Returns would/related; never mutates."""
        reason = self._validate_reason(reason)
        self._ensure_skill_exists(source)
        self._ensure_skill_exists(target)
        edge = self._normalize_edge(source, target, edge_type)
        if edge.key() in self._edge_lookup():
            raise GraphCLIError("EDGE_ALREADY_EXISTS", f"Edge already exists: {edge.key()}")
        self._assert_acyclic(edge.source, edge.target, edge_type)
        self._assert_no_contradiction(edge.source, edge.target, edge_type)
        return {
            "would": {
                "action": "add",
                "source": edge.source,
                "target": edge.target,
                "type": edge_type,
                "reason": reason,
            },
            "related": self._gather_related(edge.source, edge.target),
        }

    def propose_remove_edge(
        self, source: str, target: str, edge_type: str, reason: str
    ) -> dict[str, Any]:
        """Dry-run remove. Returns would/related; never mutates."""
        reason = self._validate_reason(reason)
        edge = self._normalize_edge(source, target, edge_type)
        if edge.key() not in self._edge_lookup():
            raise GraphCLIError("EDGE_NOT_FOUND", f"Edge does not exist: {edge.key()}")
        return {
            "would": {
                "action": "remove",
                "source": edge.source,
                "target": edge.target,
                "type": edge_type,
                "reason": reason,
            },
            "related": self._gather_related(edge.source, edge.target),
        }

    def propose_retype_edge(
        self, source: str, target: str, from_type: str, to_type: str, reason: str
    ) -> dict[str, Any]:
        """Dry-run retype. Returns would/related; never mutates."""
        reason = self._validate_reason(reason)
        if from_type == to_type:
            raise GraphCLIError("INVALID_EDGE_TYPE", "from_type and to_type must differ")
        old_edge = self._normalize_edge(source, target, from_type)
        new_edge = self._normalize_edge(source, target, to_type)
        lookup = self._edge_lookup()
        if old_edge.key() not in lookup:
            raise GraphCLIError("EDGE_NOT_FOUND", f"Edge does not exist: {old_edge.key()}")
        if new_edge.key() in lookup:
            raise GraphCLIError("EDGE_ALREADY_EXISTS", f"Edge already exists: {new_edge.key()}")
        self._assert_acyclic(new_edge.source, new_edge.target, to_type)
        self._assert_no_contradiction(
            new_edge.source,
            new_edge.target,
            to_type,
            ignore_edge_key=old_edge.key(),
        )
        return {
            "would": {
                "action": "retype",
                "source": old_edge.source,
                "target": old_edge.target,
                "from_type": from_type,
                "to_type": to_type,
                "reason": reason,
            },
            "related": self._gather_related(old_edge.source, old_edge.target),
        }

    # ------------------------------------------------------------------
    # Edit (commit) — only mutation path. No threshold, no preview loop.
    # ------------------------------------------------------------------

    def _append_history(
        self,
        action: str,
        reason: str,
        task_id: str,
        *,
        edge: dict[str, Any] | None = None,
        from_edge: dict[str, Any] | None = None,
        to_edge: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        entry: dict[str, Any] = {
            "action": action,
            "applied_at": now,
            "reason": reason,
            "online": True,
            "evidence": [{"task_id": task_id, "reason": reason, "recorded_at": now}],
        }
        if edge is not None:
            entry["edge"] = dict(edge)
        if from_edge is not None:
            entry["from_edge"] = dict(from_edge)
        if to_edge is not None:
            entry["to_edge"] = dict(to_edge)
        self.data["history"].append(entry)
        return entry

    def edit_edge(
        self,
        action: str,
        *,
        source: str,
        target: str,
        reason: str,
        task_id: str = "",
        edge_type: str | None = None,
        from_type: str | None = None,
        to_type: str | None = None,
    ) -> dict[str, Any]:
        """Commit an add / remove / retype immediately."""
        if action not in EDIT_ACTIONS:
            raise GraphCLIError(
                "INVALID_ACTION",
                f"action must be one of {sorted(EDIT_ACTIONS)}, got {action!r}",
            )
        reason = self._validate_reason(reason)

        if action == "add":
            if not edge_type:
                raise GraphCLIError("INVALID_EDGE_TYPE", "edge_type is required for add")
            self._ensure_skill_exists(source)
            self._ensure_skill_exists(target)
            edge = self._normalize_edge(source, target, edge_type)
            if edge.key() in self._edge_lookup():
                raise GraphCLIError("EDGE_ALREADY_EXISTS", f"Edge already exists: {edge.key()}")
            self._assert_acyclic(edge.source, edge.target, edge_type)
            self._assert_no_contradiction(edge.source, edge.target, edge_type)
            applied_edge = {
                **edge.as_dict(),
                "origin": "online",
                "reason": reason,
            }
            self.data["edges"].append(applied_edge)
            history_entry = self._append_history(
                "add_edge", reason, task_id, edge=applied_edge
            )
            self.save()
            return {
                "applied": True,
                "action": "add",
                "edge": applied_edge,
                "history_entry": history_entry,
            }

        if action == "remove":
            if not edge_type:
                raise GraphCLIError("INVALID_EDGE_TYPE", "edge_type is required for remove")
            edge = self._normalize_edge(source, target, edge_type)
            lookup = self._edge_lookup()
            if edge.key() not in lookup:
                raise GraphCLIError("EDGE_NOT_FOUND", f"Edge does not exist: {edge.key()}")
            removed_edge = dict(lookup[edge.key()])
            self.data["edges"] = [
                e
                for e in self.data["edges"]
                if self._normalize_edge(e["source"], e["target"], e["type"]).key() != edge.key()
            ]
            history_entry = self._append_history(
                "remove_edge", reason, task_id, edge=removed_edge
            )
            self.save()
            return {
                "applied": True,
                "action": "remove",
                "edge": removed_edge,
                "history_entry": history_entry,
            }

        # retype
        if not from_type or not to_type:
            raise GraphCLIError(
                "INVALID_EDGE_TYPE", "from_type and to_type are required for retype"
            )
        if from_type == to_type:
            raise GraphCLIError("INVALID_EDGE_TYPE", "from_type and to_type must differ")
        old = self._normalize_edge(source, target, from_type)
        new = self._normalize_edge(source, target, to_type)
        lookup = self._edge_lookup()
        if old.key() not in lookup:
            raise GraphCLIError("EDGE_NOT_FOUND", f"Edge does not exist: {old.key()}")
        if new.key() in lookup:
            raise GraphCLIError("EDGE_ALREADY_EXISTS", f"Edge already exists: {new.key()}")
        self._assert_acyclic(new.source, new.target, to_type)
        self._assert_no_contradiction(
            new.source,
            new.target,
            to_type,
            ignore_edge_key=old.key(),
        )
        previous_edge = dict(lookup[old.key()])
        self.data["edges"] = [
            e
            for e in self.data["edges"]
            if self._normalize_edge(e["source"], e["target"], e["type"]).key() != old.key()
        ]
        applied_edge = {
            **new.as_dict(),
            "origin": "online",
            "reason": reason,
        }
        self.data["edges"].append(applied_edge)
        history_entry = self._append_history(
            "retype_edge", reason, task_id, from_edge=previous_edge, to_edge=applied_edge
        )
        self.save()
        return {
            "applied": True,
            "action": "retype",
            "from_edge": previous_edge,
            "to_edge": applied_edge,
            "history_entry": history_entry,
        }

    def rollback(
        self, *, steps: int = 1, task_id: str = "", reason: str = ""
    ) -> dict[str, Any]:
        """Revert committed edit(s). Without ``task_id`` reverts the most
        recent ``steps`` reversible entries (LIFO); with ``task_id``
        reverts every entry whose evidence references that task. A
        compensating ``rollback_edge`` entry is appended so the history
        stays append-only and auditable."""
        reason = self._validate_reason(reason or "rollback")
        reversible = {"add_edge", "remove_edge", "retype_edge"}
        hist = self.data["history"]
        targets: list[int] = []
        for i in range(len(hist) - 1, -1, -1):
            e = hist[i]
            if e.get("action") not in reversible:
                continue
            if task_id:
                if task_id not in {x.get("task_id") for x in e.get("evidence", [])}:
                    continue
                targets.append(i)
            else:
                targets.append(i)
                if len(targets) >= steps:
                    break
        if not targets:
            raise GraphCLIError(
                "NOTHING_TO_ROLLBACK", "No reversible history entry matched."
            )

        def _key(ed: dict[str, Any]) -> str:
            return self._normalize_edge(ed["source"], ed["target"], ed["type"]).key()

        reverted: list[dict[str, Any]] = []
        for i in targets:  # most-recent-first
            e = hist[i]
            act = e["action"]
            if act == "add_edge":
                ed = e["edge"]
                self.data["edges"] = [
                    x for x in self.data["edges"] if _key(x) != _key(ed)
                ]
                reverted.append({"undid": "add_edge", "removed": ed})
            elif act == "remove_edge":
                ed = dict(e["edge"])
                self.data["edges"].append(ed)
                reverted.append({"undid": "remove_edge", "restored": ed})
            elif act == "retype_edge":
                to_e, from_e = e["to_edge"], dict(e["from_edge"])
                self.data["edges"] = [
                    x for x in self.data["edges"] if _key(x) != _key(to_e)
                ]
                self.data["edges"].append(from_e)
                reverted.append(
                    {"undid": "retype_edge", "from": to_e, "back_to": from_e}
                )
        entry = self._append_history(
            "rollback_edge", reason, task_id or "(last)"
        )
        entry["reverted"] = reverted
        self.save()
        return {
            "applied": True,
            "action": "rollback",
            "n_reverted": len(reverted),
            "reverted": reverted,
            "history_entry": entry,
        }
