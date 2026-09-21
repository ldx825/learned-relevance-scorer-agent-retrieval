import json
import tempfile
import unittest
from pathlib import Path

from skilldag.data_pipeline import build_action_skill_candidates


class ActionCandidateTests(unittest.TestCase):
    def test_reuses_skill_vectors_and_only_embeds_non_noop_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            task = {
                "record_id": "task:1",
                "source_task_id": "source-1",
                "split": "train",
                "task_type": "cool",
                "plan_high_pddl": [
                    {"action": "CoolObject", "args": ["apple"]},
                    {"action": "NoOp", "args": []},
                ],
            }
            tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")
            graph = root / "graph.json"
            graph.write_text(
                json.dumps(
                    {
                        "nodes": {"cooler": {}, "picker": {}, "helper": {}},
                        "edges": [
                            {"source": "cooler", "target": "helper", "type": "depends_on"}
                        ],
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )
            embeddings = root / "embeddings.json"
            embeddings.write_text(
                json.dumps(
                    {
                        "cooler": {"model": "test-model", "text_hash": "a", "embedding": [1.0, 0.0]},
                        "picker": {"model": "test-model", "text_hash": "b", "embedding": [0.0, 1.0]},
                        "helper": {"model": "test-model", "text_hash": "c", "embedding": [0.5, 0.5]},
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_embed(texts):
                calls.append(texts)
                return [[1.0, 0.0] for _ in texts]

            output = root / "output"
            report = build_action_skill_candidates(
                tasks, graph, embeddings, output, "test-model", fake_embed, top_k=1
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 1)
            self.assertIn("cool object", calls[0][0])
            self.assertEqual(report["embedding_usage"]["skill_vectors_computed"], 0)
            self.assertEqual(report["embedding_usage"]["skill_vectors_reused"], 3)
            self.assertEqual(report["embedding_usage"]["action_vectors_computed"], 1)
            self.assertEqual(report["actions"]["excluded_action_count"], 1)
            self.assertEqual(report["configuration"]["graph_seed_k"], 0)
            action_rows = [
                json.loads(line)
                for line in (output / "intermediate" / "action_skill_candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["skill_id"] for row in action_rows], ["cooler"])
            self.assertTrue(all(row["candidate_only"] for row in action_rows))


if __name__ == "__main__":
    unittest.main()
