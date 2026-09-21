import json
import os
import tempfile
import unittest
from unittest.mock import patch

from skilldag.initialize import InitializationError, _classify_anchor_batch, initialize_edges


class InitializationClassifierTests(unittest.TestCase):
    def test_classifier_maps_candidate_refs_to_skill_ids(self):
        body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [
                                {
                                    "target_ref": "B1",
                                    "type": "depends_on",
                                    "source_ref": "A",
                                    "reason": "Anchor requires candidate output.",
                                },
                                {
                                    "target_ref": "B2",
                                    "type": "specializes",
                                    "source_ref": "B2",
                                    "reason": "Candidate is the narrower skill.",
                                },
                            ]
                        )
                    }
                }
            ]
        }

        with patch.dict(os.environ, {"SKILLDAG_LLM_API_KEY": "test-key"}), patch(
            "skilldag.initialize._http_post_json",
            return_value=(200, json.dumps(body)),
        ):
            edges = _classify_anchor_batch(
                {"id": "anchor-skill", "description": "", "body_preview": ""},
                [
                    {"id": "candidate-one", "description": "", "body_preview": ""},
                    {"id": "candidate-two", "description": "", "body_preview": ""},
                ],
            )

        self.assertEqual(
            edges,
            [
                {
                    "source": "anchor-skill",
                    "target": "candidate-one",
                    "type": "depends_on",
                    "reason": "Anchor requires candidate output.",
                },
                {
                    "source": "candidate-two",
                    "target": "anchor-skill",
                    "type": "specializes",
                    "reason": "Candidate is the narrower skill.",
                },
            ],
        )

    def test_classifier_drops_cross_pair_source_refs(self):
        body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [
                                {
                                    "target_ref": "B1",
                                    "type": "depends_on",
                                    "source_ref": "B2",
                                    "reason": "Wrong candidate source.",
                                }
                            ]
                        )
                    }
                }
            ]
        }

        with patch.dict(os.environ, {"SKILLDAG_LLM_API_KEY": "test-key"}), patch(
            "skilldag.initialize._http_post_json",
            return_value=(200, json.dumps(body)),
        ):
            edges = _classify_anchor_batch(
                {"id": "anchor-skill", "description": "", "body_preview": ""},
                [
                    {"id": "candidate-one", "description": "", "body_preview": ""},
                    {"id": "candidate-two", "description": "", "body_preview": ""},
                ],
            )

        self.assertEqual(edges, [])

    def test_initialize_requires_complete_eself_embeddings_cache(self):
        nodes = {
            "skill-a": {"description": "A", "path": ""},
            "skill-b": {"description": "B", "path": ""},
        }
        cache = {
            "skill-a": {
                "model": "text-embedding-3-large",
                "embedding": [1.0, 0.0],
            }
        }

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
            json.dump(cache, f)
            f.flush()

            with self.assertRaises(InitializationError) as cm, patch(
                "skilldag.initialize._embed_batch",
                side_effect=AssertionError("e_self embedding should not run"),
            ):
                initialize_edges(nodes, embeddings_cache_path=f.name)

        self.assertIn("incomplete", str(cm.exception))
        self.assertIn("skill-b", str(cm.exception))

    def test_initialize_uses_complete_eself_cache_without_embedding_call(self):
        nodes = {
            "skill-a": {"description": "A", "path": ""},
            "skill-b": {"description": "B", "path": ""},
        }
        cache = {
            "skill-a": {
                "model": "text-embedding-3-large",
                "embedding": [1.0, 0.0],
            },
            "skill-b": {
                "model": "text-embedding-3-large",
                "embedding": [0.0, 1.0],
            },
        }

        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
            json.dump(cache, f)
            f.flush()

            with patch(
                "skilldag.initialize._embed_batch",
                side_effect=AssertionError("e_self embedding should not run"),
            ), patch(
                "skilldag.initialize._extract_imagined_needs",
                return_value="self-contained",
            ), patch("skilldag.initialize._classify_anchor_batch", return_value=[]):
                edges = initialize_edges(nodes, embeddings_cache_path=f.name)

        self.assertEqual(edges, [])


if __name__ == "__main__":
    unittest.main()
