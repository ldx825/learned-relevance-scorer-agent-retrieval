import json
import tempfile
import unittest
from pathlib import Path

from skilldag.data_pipeline import run_judge_pilot


class JudgePilotTests(unittest.TestCase):
    def test_validates_and_caches_structured_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "record_id": "task:1",
                        "task_type": "cool",
                        "task_text": "Cool an apple.",
                        "plan_high_pddl": [{"action": "CoolObject", "args": ["apple"]}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            skills = root / "skills.jsonl"
            skills.write_text(
                json.dumps(
                    {
                        "skill_id": "cooler",
                        "description": "Cools objects.",
                        "skill_body": "Use a refrigerator.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidates = root / "candidates.jsonl"
            candidates.write_text(
                json.dumps(
                    {
                        "task_record_id": "task:1",
                        "skill_id": "cooler",
                        "candidate_sources": ["task_embedding_match"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selection = root / "selection.json"
            selection.write_text(json.dumps({"record_ids": ["task:1"]}), encoding="utf-8")
            calls = []

            def chat(payload):
                calls.append(payload)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "judgments": [
                                            {
                                                "skill_id": "cooler",
                                                "necessity": "required",
                                                "relevance": 0.98,
                                                "confidence": 0.9,
                                                "reason": "The plan explicitly cools the object.",
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }

            output = root / "output"
            first = run_judge_pilot(tasks, skills, candidates, selection, output, "test-model", chat)
            self.assertEqual(first["successful_task_count"], 1)
            self.assertEqual(first["label_counts"], {"required": 1})
            self.assertEqual(first["final_training_labels_generated"], 0)
            self.assertEqual(len(calls), 1)

            def no_api(_):
                raise AssertionError("successful cache should be reused")

            second = run_judge_pilot(tasks, skills, candidates, selection, output, "test-model", no_api)
            self.assertEqual(second["cache_hit_task_count"], 1)
            evidence = json.loads((output / "labels" / "pilot_judge_evidence.jsonl").read_text())
            self.assertEqual(evidence["raw_label"], "required")
            self.assertTrue(evidence["expert_plan_visible"])

    def test_named_collection_does_not_overwrite_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            tasks.write_text(json.dumps({"record_id": "t", "task_text": "do it"}) + "\n")
            skills = root / "skills.jsonl"
            skills.write_text(
                json.dumps({"skill_id": "s", "description": "do", "skill_body": "do"}) + "\n"
            )
            candidates = root / "candidates.jsonl"
            candidates.write_text(json.dumps({"task_record_id": "t", "skill_id": "s"}) + "\n")
            selection = root / "selection.json"
            selection.write_text(json.dumps({"record_ids": ["t"]}))

            def chat(_payload):
                return {"choices": [{"message": {"content": json.dumps({"judgments": [{
                    "skill_id": "s", "necessity": "helpful", "relevance": 0.7,
                    "confidence": 0.8, "reason": "useful"
                }]})}}]}

            report = run_judge_pilot(
                tasks, skills, candidates, selection, root / "out", "m", chat,
                collection_name="medium_v1",
            )
            self.assertEqual(report["collection_name"], "medium_v1")
            self.assertTrue((root / "out/labels/medium_v1_judge_evidence.jsonl").exists())
            self.assertFalse((root / "out/labels/pilot_judge_evidence.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
