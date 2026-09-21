import json
import tempfile
import unittest
from pathlib import Path

from skilldag.data_pipeline import evaluate_reranking_baselines


class RerankingBenchmarkTests(unittest.TestCase):
    def test_task_cosine_rewards_required_skill_and_clean_filter_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "record_id": "task:1",
                        "task_type": "pick_heat_then_place_in_recep",
                        "task_text": "Heat an apple.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidates = root / "candidates.jsonl"
            candidate_rows = [
                {
                    "task_record_id": "task:1",
                    "skill_id": "heater",
                    "task_cosine_score": 0.9,
                    "action_cosine_score": 0.8,
                    "task_semantic_rank": 1,
                    "action_semantic_rank": 1,
                },
                {
                    "task_record_id": "task:1",
                    "skill_id": "cooler",
                    "task_cosine_score": 0.4,
                    "action_cosine_score": 0.7,
                    "task_semantic_rank": 2,
                    "action_semantic_rank": 2,
                },
            ]
            candidates.write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8"
            )
            evidence = root / "evidence.jsonl"
            evidence_rows = [
                {
                    "task_record_id": "task:1",
                    "skill_id": "heater",
                    "raw_label": "required",
                    "reason": "The task needs heating.",
                },
                {
                    "task_record_id": "task:1",
                    "skill_id": "cooler",
                    "raw_label": "irrelevant",
                    "reason": "Cooling is the opposite action.",
                },
            ]
            evidence.write_text(
                "".join(json.dumps(row) + "\n" for row in evidence_rows), encoding="utf-8"
            )
            graph = root / "graph.json"
            graph.write_text(json.dumps({"nodes": {}, "edges": []}), encoding="utf-8")
            report = evaluate_reranking_baselines(tasks, candidates, evidence, graph, root / "out")
            task_metrics = report["subsets"]["all_weak"]["methods"]["task_cosine"]
            self.assertEqual(task_metrics["ndcg@3"], 1.0)
            self.assertEqual(task_metrics["required_recall@3"], 1.0)
            self.assertEqual(report["subsets"]["clean"]["task_count"], 1)
            self.assertFalse(report["judge_evidence_is_gold"])


if __name__ == "__main__":
    unittest.main()
