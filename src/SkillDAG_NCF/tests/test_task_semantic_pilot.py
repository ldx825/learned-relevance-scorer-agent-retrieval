import json
import tempfile
import unittest
from pathlib import Path

from skilldag.data_pipeline import build_task_semantic_pilot


class TaskSemanticPilotTests(unittest.TestCase):
    def test_stratified_task_candidates_merge_with_action_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            task_rows = [
                {
                    "record_id": "task:cool",
                    "source_task_id": "cool",
                    "split": "train",
                    "task_type": "cool",
                    "task_text": "Cool the apple.",
                    "plan_high_pddl": [],
                },
                {
                    "record_id": "task:heat",
                    "source_task_id": "heat",
                    "split": "train",
                    "task_type": "heat",
                    "task_text": "Heat the apple.",
                    "plan_high_pddl": [],
                },
            ]
            tasks.write_text("".join(json.dumps(row) + "\n" for row in task_rows), encoding="utf-8")
            action_candidates = root / "action.jsonl"
            action_candidates.write_text(
                json.dumps(
                    {
                        "task_record_id": "task:cool",
                        "skill_id": "helper",
                        "candidate_sources": ["action_embedding_match"],
                        "best_cosine_score": 0.4,
                        "best_semantic_rank": 1,
                        "matched_actions": ["CoolObject"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            graph = root / "graph.json"
            graph.write_text(json.dumps({"nodes": {"cooler": {}, "heater": {}, "helper": {}}}), encoding="utf-8")
            skill_embeddings = root / "skills.json"
            skill_embeddings.write_text(
                json.dumps(
                    {
                        "cooler": {"model": "test", "embedding": [1.0, 0.0]},
                        "heater": {"model": "test", "embedding": [0.0, 1.0]},
                        "helper": {"model": "test", "embedding": [0.5, 0.5]},
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_embed(texts):
                calls.append(texts)
                return [[1.0, 0.0], [0.0, 1.0]]

            output = root / "output"
            report = build_task_semantic_pilot(
                tasks,
                action_candidates,
                graph,
                skill_embeddings,
                output,
                "test",
                fake_embed,
                samples_per_task_type=1,
                task_top_k=1,
            )
            self.assertEqual(report["sample"]["task_count"], 2)
            self.assertEqual(report["embedding_usage"]["task_vectors_computed"], 2)
            self.assertEqual(report["candidates"]["task_semantic_pair_count"], 2)
            self.assertEqual(report["candidates"]["combined_pair_count"], 3)
            self.assertEqual(report["skill_coverage"]["combined_count"], 3)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
