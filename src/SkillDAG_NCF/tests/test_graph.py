import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skilldag.graph import GraphCLIError, SkillGraph


class SkillGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.skills_dir = self.root / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.skills_dir / "skillgraph.json"

        self._create_skill("skill-a", "Skill A", "Primary skill A")
        self._create_skill("skill-b", "Skill B", "Dependency skill B")
        self._create_skill("skill-c", "Skill C", "Conflicting skill C")
        self._create_skill("skill-d", "Skill D", "Alternative skill D")
        self._create_skill("skill-e", "Skill E", "Specialized skill E")
        self.graph_path.write_text(
            json.dumps(
                {
                    "schema_version": "skillgraph.v1",
                    "updated_at": "2026-04-20T00:00:00Z",
                    "nodes": {},
                    "edges": [],
                    "history": [],
                }
            ),
            encoding="utf-8",
        )

        self.graph = SkillGraph.load(graph_path=self.graph_path, skills_dir=self.skills_dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_skill(self, skill_id: str, name: str, description: str):
        skill_dir = self.skills_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
            encoding="utf-8",
        )

    def _reload(self):
        self.graph = SkillGraph.load(graph_path=self.graph_path, skills_dir=self.skills_dir)
        return self.graph

    def _patch_embedding_backend(self, graph):
        """Install a deterministic embedding backend for offline search tests."""
        vocab = (
            "alpha", "keyword", "matcher", "beta", "chunker", "gamma",
            "aggregator", "zeta", "remote", "root", "main", "retrieval",
            "target", "solo", "query",
        )

        def embed(text: str) -> list[float]:
            text = text.lower()
            return [float(text.count(term)) for term in vocab]

        graph._embed_query = embed
        graph._load_or_build_node_embeddings = lambda: {
            sid: embed(
                f"{sid} {node.get('name', '')} {node.get('description', '')}"
            )
            for sid, node in graph.data["nodes"].items()
        }
        return graph

    def test_initialize_discovers_nodes(self):
        self.assertEqual(set(self.graph.data["nodes"].keys()), {"skill-a", "skill-b", "skill-c", "skill-d", "skill-e"})

    def test_ncf_search_preserves_one_proactive_plan_skill(self):
        graph = self._patch_embedding_backend(self._reload())
        query_embedding = graph._embed_query("alpha query")
        node_embeddings = graph._load_or_build_node_embeddings()
        cosine_ranking = [
            ("skill-b", 0.95),
            ("skill-c", 0.90),
            ("skill-d", 0.85),
            ("skill-e", 0.80),
            ("skill-a", 0.10),
        ]
        graph._vector_score_context = lambda _query: (
            list(cosine_ranking),
            query_embedding,
            node_embeddings,
        )
        (self.graph_path.parent / "ncf_plan.json").write_text(
            json.dumps(
                {
                    "views": [
                        {
                            "name": "full_task",
                            "matches": [{"skill_id": "skill-a"}],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        class PopularityBiasedReranker:
            @staticmethod
            def score_by_id(_query_embedding, skill_ids):
                scores = {
                    "skill-a": -10.0,
                    "skill-b": 0.0,
                    "skill-c": 100.0,
                    "skill-d": 90.0,
                    "skill-e": 80.0,
                }
                return [scores[skill_id] for skill_id in skill_ids]

        with (
            patch.dict(
                os.environ,
                {
                    "SKILLDAG_NCF_MODEL": str(self.root / "fake-model.pt"),
                    "SKILLDAG_NCF_CANDIDATE_K": "5",
                },
                clear=False,
            ),
            patch(
                "skilldag.ncf_reranker.load_reranker",
                return_value=PopularityBiasedReranker(),
            ),
        ):
            result = graph.search("alpha query", top_k=2, depth=0)

        matches = result["matches"]
        self.assertIn("skill-a", [match["skill_id"] for match in matches])
        planned = next(match for match in matches if match["skill_id"] == "skill-a")
        self.assertTrue(planned["plan_safe"])
        self.assertEqual(planned["score_type"], "calibrated_hybrid")

    def test_ncf_search_uses_full_task_phase_context(self):
        graph = self._reload()
        (self.graph_path.parent / "ncf_plan.json").write_text(
            json.dumps({"full_task": "Build the requested report.", "views": []}),
            encoding="utf-8",
        )

        with patch.dict(
            os.environ,
            {"SKILLDAG_NCF_MODEL": str(self.root / "fake-model.pt")},
            clear=False,
        ):
            context = graph._ncf_context_query("validate the output")

        self.assertEqual(
            context,
            "Full task:\nBuild the requested report.\n\n"
            "Current phase/query:\nvalidate the output",
        )

    def test_ncf_search_appends_one_audit_record_per_query(self):
        graph = self._patch_embedding_backend(self._reload())
        audit_path = self.root / "ncf_search_audit.jsonl"

        def rerank(_query, cosine_ranking, **_kwargs):
            scores = {skill_id: float(index) for index, (skill_id, _) in enumerate(cosine_ranking)}
            return list(cosine_ranking), scores, set()

        graph._ncf_rerank = rerank
        with patch.dict(
            os.environ,
            {
                "SKILLDAG_NCF_MODEL": str(self.root / "fake-model.pt"),
                "SKILLDAG_NCF_CANDIDATE_K": "100",
                "SKILLDAG_NCF_AUDIT_LOG": str(audit_path),
            },
            clear=False,
        ):
            graph.search("alpha query", top_k=5, depth=0)

        records = [json.loads(line) for line in audit_path.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["ncf_enabled"])
        self.assertTrue(records[0]["ncf_reranked"])
        self.assertEqual(records[0]["candidate_k"], 100)
        self.assertEqual(records[0]["top_k"], 5)
        self.assertNotIn("query", records[0])

    def test_load_refreshes_stale_node_paths_from_skills_dir(self):
        old_path = "/tmp/nonportable/skill-a"
        raw = json.loads(self.graph_path.read_text(encoding="utf-8"))
        raw["nodes"]["skill-a"]["path"] = old_path
        self.graph_path.write_text(json.dumps(raw), encoding="utf-8")

        graph = self._reload()

        self.assertEqual(graph.data["nodes"]["skill-a"]["path"], str(self.skills_dir / "skill-a"))
        self.assertNotEqual(graph.data["nodes"]["skill-a"]["path"], old_path)

    def test_legacy_proposals_field_is_stripped_on_load(self):
        self.graph_path.write_text(
            json.dumps(
                {
                    "schema_version": "skillgraph.v1",
                    "updated_at": "2026-04-20T00:00:00Z",
                    "nodes": {},
                    "edges": [],
                    "proposals": [{"proposal_id": "legacy", "action": "add_edge", "edge": {}, "evidence": []}],
                    "history": [],
                }
            ),
            encoding="utf-8",
        )
        graph = self._reload()
        self.assertNotIn("proposals", graph.data)
        raw = json.loads(self.graph_path.read_text(encoding="utf-8"))
        self.assertNotIn("proposals", raw)

    def test_read_ops_cover_dependencies_conflicts_and_redundancy(self):
        self.graph.data["edges"] = [
            {"source": "skill-a", "target": "skill-b", "type": "depends_on"},
            {"source": "skill-a", "target": "skill-c", "type": "conflicts_with"},
            {"source": "skill-a", "target": "skill-d", "type": "similar_to"},
            {"source": "skill-e", "target": "skill-a", "type": "specializes"},
        ]
        self.graph.save()
        graph = self._reload()

        deps = graph.get_dependencies("skill-a")
        self.assertEqual(deps["dependencies"], [{"target": "skill-b", "type": "depends_on"}])

        expanded = graph.expand_set(["skill-a"])
        self.assertEqual(expanded["added"], ["skill-b"])

        conflicts = graph.get_conflicts("skill-a")
        self.assertEqual(conflicts["conflicts"], [{"target": "skill-c", "type": "conflicts_with"}])

        alternatives = graph.get_alternatives("skill-a")
        self.assertEqual(
            alternatives["alternatives"],
            [
                {"target": "skill-d", "via": "similar_to"},
                {"target": "skill-e", "via": "specializes_child"},
            ],
        )

        report = graph.check_set(["skill-a", "skill-c", "skill-d"])
        self.assertEqual(report["missing_deps"], ["skill-b"])
        self.assertEqual(report["conflicts"], [{"source": "skill-a", "target": "skill-c", "type": "conflicts_with"}])
        self.assertEqual(report["redundant"], [{"pair": ["skill-a", "skill-d"], "reason": "similar_to"}])

    # ------------------------------------------------------------------
    # propose-* (dry-run)
    # ------------------------------------------------------------------

    def test_propose_edge_returns_empty_related_when_no_prior(self):
        graph = self._reload()
        result = graph.propose_edge(
            "skill-a", "skill-b", "depends_on", reason="planning before commit"
        )
        self.assertEqual(result["would"], {
            "action": "add",
            "source": "skill-a",
            "target": "skill-b",
            "type": "depends_on",
            "reason": "planning before commit",
        })
        self.assertEqual(result["related"], [])
        # Dry-run: graph is unchanged on disk.
        self.assertEqual(graph.data["edges"], [])

    def test_propose_retype_returns_prior_edge_reason_in_related(self):
        self.graph.data["edges"] = [
            {
                "source": "skill-a",
                "target": "skill-b",
                "type": "depends_on",
                "origin": "online",
                "reason": "prior evidence from task T",
            }
        ]
        self.graph.save()
        graph = self._reload()

        # Retyping to conflicts_with must surface the existing depends_on edge
        # and its reason while avoiding a final contradictory state.
        result = graph.propose_retype_edge(
            "skill-a",
            "skill-b",
            "depends_on",
            "conflicts_with",
            reason="probe prior pair",
        )
        kinds = [r["kind"] for r in result["related"]]
        self.assertIn("edge", kinds)
        edge_entry = next(r for r in result["related"] if r["kind"] == "edge")
        self.assertEqual(edge_entry["type"], "depends_on")
        self.assertEqual(edge_entry["reason"], "prior evidence from task T")
        self.assertEqual(edge_entry["origin"], "online")

    def test_propose_edge_surfaces_history_reasons_for_pair(self):
        graph = self._reload()
        graph.edit_edge(
            "add",
            source="skill-a",
            target="skill-b",
            edge_type="depends_on",
            reason="initial attempt",
            task_id="task-1",
        )
        graph.edit_edge(
            "remove",
            source="skill-a",
            target="skill-b",
            edge_type="depends_on",
            reason="removed after revisit",
            task_id="task-1",
        )
        graph = self._reload()

        result = graph.propose_edge(
            "skill-a", "skill-b", "conflicts_with", reason="probe history"
        )
        history_entries = [r for r in result["related"] if r["kind"] == "history"]
        reasons = [r["reason"] for r in history_entries]
        self.assertIn("initial attempt", reasons)
        self.assertIn("removed after revisit", reasons)

    def test_propose_edge_rejects_duplicate(self):
        self.graph.data["edges"] = [{"source": "skill-a", "target": "skill-b", "type": "depends_on"}]
        self.graph.save()
        graph = self._reload()
        with self.assertRaises(GraphCLIError) as ctx:
            graph.propose_edge("skill-a", "skill-b", "depends_on", reason="dup probe")
        self.assertEqual(ctx.exception.code, "EDGE_ALREADY_EXISTS")

    def test_propose_remove_requires_existing_edge(self):
        graph = self._reload()
        with self.assertRaises(GraphCLIError) as ctx:
            graph.propose_remove_edge(
                "skill-a", "skill-b", "depends_on", reason="missing probe"
            )
        self.assertEqual(ctx.exception.code, "EDGE_NOT_FOUND")

    def test_propose_retype_rejects_identical_types(self):
        self.graph.data["edges"] = [{"source": "skill-a", "target": "skill-b", "type": "depends_on"}]
        self.graph.save()
        graph = self._reload()
        with self.assertRaises(GraphCLIError) as ctx:
            graph.propose_retype_edge(
                "skill-a", "skill-b", "depends_on", "depends_on", reason="same-type probe"
            )
        self.assertEqual(ctx.exception.code, "INVALID_EDGE_TYPE")

    def test_propose_edge_requires_non_empty_reason(self):
        graph = self._reload()
        with self.assertRaises(GraphCLIError) as ctx:
            graph.propose_edge("skill-a", "skill-b", "depends_on", reason="   ")
        self.assertEqual(ctx.exception.code, "INVALID_REASON")

    def test_propose_remove_requires_non_empty_reason(self):
        self.graph.data["edges"] = [{"source": "skill-a", "target": "skill-b", "type": "depends_on"}]
        self.graph.save()
        graph = self._reload()
        with self.assertRaises(GraphCLIError) as ctx:
            graph.propose_remove_edge("skill-a", "skill-b", "depends_on", reason="")
        self.assertEqual(ctx.exception.code, "INVALID_REASON")

    def test_propose_retype_requires_non_empty_reason(self):
        self.graph.data["edges"] = [{"source": "skill-a", "target": "skill-b", "type": "depends_on"}]
        self.graph.save()
        graph = self._reload()
        with self.assertRaises(GraphCLIError) as ctx:
            graph.propose_retype_edge(
                "skill-a", "skill-b", "depends_on", "conflicts_with", reason=""
            )
        self.assertEqual(ctx.exception.code, "INVALID_REASON")

    # ------------------------------------------------------------------
    # edit-edge (commit)
    # ------------------------------------------------------------------

    def test_edit_edge_add_applies_immediately(self):
        graph = self._reload()
        result = graph.edit_edge(
            "add",
            source="skill-a",
            target="skill-b",
            edge_type="depends_on",
            reason="needed in task T",
            task_id="task-T",
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["action"], "add")
        self.assertEqual(result["edge"]["type"], "depends_on")
        self.assertEqual(result["edge"]["reason"], "needed in task T")
        self.assertEqual(result["edge"]["origin"], "online")

        graph = self._reload()
        self.assertEqual(len(graph.data["edges"]), 1)
        self.assertEqual(graph.data["edges"][0]["source"], "skill-a")
        self.assertEqual(graph.data["history"][0]["action"], "add_edge")
        self.assertEqual(graph.data["history"][0]["evidence"][0]["task_id"], "task-T")

    def test_edit_edge_remove_applies_immediately(self):
        self.graph.data["edges"] = [{"source": "skill-a", "target": "skill-b", "type": "depends_on"}]
        self.graph.save()
        graph = self._reload()

        result = graph.edit_edge(
            "remove",
            source="skill-a",
            target="skill-b",
            edge_type="depends_on",
            reason="bad edge",
            task_id="task-T",
        )
        self.assertTrue(result["applied"])

        graph = self._reload()
        self.assertEqual(graph.data["edges"], [])
        self.assertEqual(graph.data["history"][0]["action"], "remove_edge")
        self.assertEqual(graph.data["history"][0]["reason"], "bad edge")

    def test_edit_edge_retype_preserves_reason_and_history(self):
        self.graph.data["edges"] = [
            {
                "source": "skill-a",
                "target": "skill-b",
                "type": "similar_to",
                "origin": "cold_start",
                "reason": "static guess",
            }
        ]
        self.graph.save()
        graph = self._reload()

        result = graph.edit_edge(
            "retype",
            source="skill-a",
            target="skill-b",
            from_type="similar_to",
            to_type="conflicts_with",
            reason="observed co-use harm",
            task_id="task-T",
        )
        self.assertTrue(result["applied"])

        graph = self._reload()
        self.assertEqual(graph.data["edges"][0]["type"], "conflicts_with")
        self.assertEqual(graph.data["edges"][0]["origin"], "online")
        self.assertEqual(graph.data["edges"][0]["reason"], "observed co-use harm")
        self.assertEqual(graph.data["history"][0]["action"], "retype_edge")
        self.assertEqual(graph.data["history"][0]["from_edge"]["reason"], "static guess")
        self.assertEqual(graph.data["history"][0]["to_edge"]["reason"], "observed co-use harm")

    def test_edit_edge_add_rejects_duplicate(self):
        self.graph.data["edges"] = [{"source": "skill-a", "target": "skill-b", "type": "depends_on"}]
        self.graph.save()
        graph = self._reload()
        with self.assertRaises(GraphCLIError) as ctx:
            graph.edit_edge(
                "add",
                source="skill-a",
                target="skill-b",
                edge_type="depends_on",
                reason="duplicate attempt",
                task_id="task-T",
            )
        self.assertEqual(ctx.exception.code, "EDGE_ALREADY_EXISTS")

    def test_edit_edge_remove_rejects_missing_edge(self):
        graph = self._reload()
        with self.assertRaises(GraphCLIError) as ctx:
            graph.edit_edge(
                "remove",
                source="skill-a",
                target="skill-b",
                edge_type="depends_on",
                reason="attempt",
                task_id="task-T",
            )
        self.assertEqual(ctx.exception.code, "EDGE_NOT_FOUND")

    def test_edit_edge_requires_non_empty_reason(self):
        graph = self._reload()
        with self.assertRaises(GraphCLIError) as ctx:
            graph.edit_edge(
                "add",
                source="skill-a",
                target="skill-b",
                edge_type="depends_on",
                reason="   ",
                task_id="task-T",
            )
        self.assertEqual(ctx.exception.code, "INVALID_REASON")

    def test_edit_edge_accepts_empty_task_id(self):
        graph = self._reload()
        result = graph.edit_edge(
            "add",
            source="skill-a",
            target="skill-b",
            edge_type="depends_on",
            reason="needed",
        )
        self.assertTrue(result["applied"])
        graph = self._reload()
        self.assertEqual(graph.data["history"][0]["evidence"][0]["task_id"], "")

    def test_edit_edge_can_commit_despite_prior_edge_on_same_pair(self):
        """Different edge types on the same pair co-exist — the propose step
        surfaces the prior reason, but edit-edge still commits when the agent
        explicitly asks for a different type."""
        self.graph.data["edges"] = [
            {
                "source": "skill-a",
                "target": "skill-b",
                "type": "depends_on",
                "origin": "online",
                "reason": "earlier reasoning",
            }
        ]
        self.graph.save()
        graph = self._reload()

        # Agent calls propose_retype first, reads the related depends_on
        # reason, then commits the coherent retype.
        preview = graph.propose_retype_edge(
            "skill-a",
            "skill-b",
            "depends_on",
            "conflicts_with",
            reason="check before retype",
        )
        self.assertTrue(any(r["type"] == "depends_on" for r in preview["related"]))

        # Retype is the coherent move, not add.
        result = graph.edit_edge(
            "retype",
            source="skill-a",
            target="skill-b",
            from_type="depends_on",
            to_type="conflicts_with",
            reason="co-use harm seen twice in task T",
            task_id="task-T",
        )
        self.assertTrue(result["applied"])

        graph = self._reload()
        self.assertEqual(len(graph.data["edges"]), 1)
        self.assertEqual(graph.data["edges"][0]["type"], "conflicts_with")

    # ------------------------------------------------------------------
    # search / BFS coverage — unchanged
    # ------------------------------------------------------------------

    def test_search_matches_vs_neighbors_no_mixing(self):
        self._create_skill("alpha-finder", "Alpha Finder", "keyword matcher for alpha corpus")
        self._create_skill("beta-chunker", "Beta Chunker", "word splitter utility")
        self._create_skill("gamma-aggregator", "Gamma Aggregator", "combines ranked outputs")
        self._create_skill("delta-clone", "Delta Clone", "redundant backend discarded on selection")
        self._create_skill("zeta-remote", "Zeta Remote", "unrelated remote runner")
        graph = self._reload()
        graph.data["edges"] = [
            {"source": "alpha-finder", "target": "beta-chunker", "type": "depends_on"},
            {"source": "alpha-finder", "target": "gamma-aggregator", "type": "composes_with"},
            {"source": "alpha-finder", "target": "delta-clone", "type": "conflicts_with"},
            {"source": "gamma-aggregator", "target": "zeta-remote", "type": "similar_to"},
        ]
        graph.save()
        graph = self._patch_embedding_backend(self._reload())

        query = "alpha keyword matcher"

        no_nbr = graph.search(query, top_k=1, depth=0)
        match_ids = [m["skill_id"] for m in no_nbr["matches"]]
        self.assertIn("alpha-finder", match_ids)
        self.assertNotIn("beta-chunker", match_ids)
        self.assertNotIn("gamma-aggregator", match_ids)
        self.assertEqual(no_nbr["neighbors"], [])

        one = graph.search(query, top_k=1, depth=1)
        one_match_ids = [m["skill_id"] for m in one["matches"]]
        self.assertIn("alpha-finder", one_match_ids)
        self.assertNotIn("beta-chunker", one_match_ids)
        self.assertNotIn("gamma-aggregator", one_match_ids)
        one_nbr = {n["skill_id"]: n for n in one["neighbors"]}
        self.assertEqual(one_nbr["beta-chunker"]["depth"], 1)
        self.assertEqual(one_nbr["beta-chunker"]["reached_from"], "alpha-finder")
        self.assertEqual(one_nbr["beta-chunker"]["via"], "depends_on")
        self.assertEqual(one_nbr["gamma-aggregator"]["via"], "composes_with")
        self.assertNotIn("delta-clone", one_nbr)
        self.assertNotIn("zeta-remote", one_nbr)

        two = graph.search(query, top_k=1, depth=2)
        two_nbr = {n["skill_id"]: n for n in two["neighbors"]}
        self.assertEqual(two_nbr["zeta-remote"]["depth"], 2)
        self.assertEqual(two_nbr["zeta-remote"]["reached_from"], "gamma-aggregator")
        self.assertEqual(two_nbr["zeta-remote"]["via"], "similar_to")
        self.assertNotIn("delta-clone", two_nbr)
        self.assertNotIn("alpha-finder", two_nbr)

    def test_search_specializes_edge_is_walkable_both_directions(self):
        self._create_skill("root-finder", "Root Finder", "query root token matcher")
        self._create_skill("narrow-variant", "Narrow Variant", "narrower specialization instance")
        self._create_skill("broad-umbrella", "Broad Umbrella", "broader unification of variants")
        graph = self._reload()
        graph.data["edges"] = [
            {"source": "root-finder", "target": "broad-umbrella", "type": "specializes"},
            {"source": "narrow-variant", "target": "root-finder", "type": "specializes"},
        ]
        graph.save()
        graph = self._patch_embedding_backend(self._reload())

        r = graph.search("root", top_k=1, depth=1)
        match_ids = [m["skill_id"] for m in r["matches"]]
        self.assertEqual(match_ids, ["root-finder"])
        nbr = {n["skill_id"]: n for n in r["neighbors"]}
        self.assertEqual(nbr["broad-umbrella"]["via"], "specializes")
        self.assertEqual(nbr["broad-umbrella"]["depth"], 1)
        self.assertEqual(nbr["narrow-variant"]["via"], "specializes")
        self.assertEqual(nbr["narrow-variant"]["depth"], 1)

    def test_search_conflicts_section_surfaces_conflict_edges(self):
        self._create_skill("match-primary", "Match Primary", "main retrieval target")
        self._create_skill("conflict-duplicate", "Conflict Duplicate", "separate functionality, no overlap")
        self._create_skill("bridge-neighbor", "Bridge Neighbor", "reachable via specializes")
        graph = self._reload()
        graph.data["edges"] = [
            {
                "source": "match-primary",
                "target": "conflict-duplicate",
                "type": "conflicts_with",
                "origin": "online",
                "reason": "co-load triggers duplicate state writes in task T-42",
            },
            {"source": "match-primary", "target": "bridge-neighbor", "type": "specializes"},
        ]
        graph.save()
        graph = self._patch_embedding_backend(self._reload())

        r = graph.search("main retrieval target", top_k=1, depth=1)

        match_ids = [m["skill_id"] for m in r["matches"]]
        self.assertIn("match-primary", match_ids)
        self.assertNotIn("conflict-duplicate", match_ids)

        nbr_ids = [n["skill_id"] for n in r["neighbors"]]
        self.assertIn("bridge-neighbor", nbr_ids)
        self.assertNotIn("conflict-duplicate", nbr_ids)

        conf = r["conflicts"]
        self.assertEqual(len(conf), 1)
        self.assertEqual(conf[0]["skill_id"], "conflict-duplicate")
        self.assertEqual(conf[0]["conflicts_with"], "match-primary")
        self.assertEqual(conf[0]["reason"], "co-load triggers duplicate state writes in task T-42")
        self.assertEqual(conf[0]["origin"], "online")

    def test_search_conflicts_section_is_empty_when_no_conflict_edges(self):
        self._create_skill("solo-match", "Solo Match", "isolated skill")
        graph = self._patch_embedding_backend(self._reload())
        r = graph.search("solo", top_k=1, depth=1)
        self.assertEqual(r["conflicts"], [])

    def test_search_matches_are_purely_lexical(self):
        self._create_skill("query-root", "Query Root", "query and retrieval root")
        self._create_skill("hidden-helper", "Hidden Helper", "no shared tokens at all")
        graph = self._reload()
        graph.data["edges"] = [
            {"source": "query-root", "target": "hidden-helper", "type": "composes_with"},
        ]
        graph.save()
        graph = self._patch_embedding_backend(self._reload())

        r = graph.search("query retrieval", top_k=1, depth=3)
        match_ids = [m["skill_id"] for m in r["matches"]]
        self.assertIn("query-root", match_ids)
        self.assertNotIn("hidden-helper", match_ids)
        nbr_ids = [n["skill_id"] for n in r["neighbors"]]
        self.assertIn("hidden-helper", nbr_ids)

    # ------------------------------------------------------------------
    # CLI smoke
    # ------------------------------------------------------------------

    def test_cli_graph_get_skill_outputs_json(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "skilldag",
                "graph",
                "--graph-path",
                str(self.graph_path),
                "--skills-dir",
                str(self.skills_dir),
                "get-skill",
                "skill-a",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["op"], "get-skill")
        self.assertEqual(payload["result"]["skill_id"], "skill-a")

    def test_cli_graph_uses_env_default_paths(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        env["SKILLDAG_SKILLS_DIR"] = str(self.skills_dir)
        env["SKILLDAG_GRAPH_PATH"] = str(self.graph_path)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "skilldag",
                "graph",
                "get-skill",
                "skill-b",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["skill_id"], "skill-b")

    def test_cli_graph_propose_edge_returns_related(self):
        self.graph.data["edges"] = [{
            "source": "skill-a",
            "target": "skill-b",
            "type": "depends_on",
            "origin": "online",
            "reason": "prior reason",
        }]
        self.graph.save()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        proc = subprocess.run(
            [
                sys.executable, "-m", "skilldag", "graph",
                "--graph-path", str(self.graph_path),
                "--skills-dir", str(self.skills_dir),
                "propose-retype", "skill-a", "skill-b",
                "--from", "depends_on", "--to", "conflicts_with",
                "--reason", "cli probe",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["op"], "propose-retype")
        self.assertEqual(payload["result"]["would"]["to_type"], "conflicts_with")
        self.assertTrue(any(r.get("type") == "depends_on" and r.get("reason") == "prior reason"
                            for r in payload["result"]["related"]))

    def test_cli_graph_edit_edge_add_commits(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        proc = subprocess.run(
            [
                sys.executable, "-m", "skilldag", "graph",
                "--graph-path", str(self.graph_path),
                "--skills-dir", str(self.skills_dir),
                "edit-edge", "add", "skill-a", "skill-b", "depends_on",
                "--reason", "cli add test",
                "--task-id", "task-T",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["op"], "edit-edge")
        self.assertTrue(payload["result"]["applied"])

        graph = self._reload()
        self.assertEqual(len(graph.data["edges"]), 1)
        self.assertEqual(graph.data["edges"][0]["type"], "depends_on")


if __name__ == "__main__":
    unittest.main()
