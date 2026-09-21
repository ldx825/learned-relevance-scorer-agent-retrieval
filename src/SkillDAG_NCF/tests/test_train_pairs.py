import json
import tempfile
import unittest
from pathlib import Path

from skilldag.data_pipeline import build_task_skill_dataset


class TrainPairsTests(unittest.TestCase):
    def test_builds_auditable_pairs_without_text_group_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            task_rows = []
            for index in range(10):
                task_rows.append({
                    "record_id": f"t{index}", "task_type": "type", "split": "train",
                    "task_text": f"task {index}", "task_text_hash": f"h{index}",
                })
            tasks.write_text("".join(json.dumps(row) + "\n" for row in task_rows))
            skills = root / "skills.jsonl"
            skills.write_text("".join(json.dumps({"skill_id": sid, "skill_text": sid}) + "\n" for sid in ("s1", "s2")))
            candidates = root / "candidates.jsonl"
            evidence = root / "evidence.jsonl"
            candidate_rows, evidence_rows = [], []
            for index in range(10):
                for rank, sid in enumerate(("s1", "s2"), 1):
                    candidate_rows.append({
                        "task_record_id": f"t{index}", "skill_id": sid,
                        "candidate_sources": ["task_embedding_match"],
                        "task_semantic_rank": rank, "task_cosine_score": 1 / rank,
                    })
                    evidence_rows.append({
                        "evidence_id": f"e{index}{sid}", "task_record_id": f"t{index}",
                        "skill_id": sid, "raw_label": "required" if sid == "s1" else "irrelevant",
                        "score": 0.9 if sid == "s1" else 0.1, "confidence": 0.8,
                        "label_source": "llm_judge",
                    })
            # The production candidate file covers all train tasks, while the
            # Judge evidence intentionally covers only the selected subset.
            candidate_rows.append({"task_record_id": "outside-selection", "skill_id": "s1"})
            candidates.write_text("".join(json.dumps(row) + "\n" for row in candidate_rows))
            evidence.write_text("".join(json.dumps(row) + "\n" for row in evidence_rows))
            graph = root / "graph.json"
            graph.write_text(json.dumps({"nodes": {"s1": {}, "s2": {}}, "edges": [{"source": "s1", "target": "s2", "type": "similar_to"}]}))
            pilot = root / "pilot.json"
            pilot.write_text(json.dumps({"record_ids": ["t0"]}))

            report = build_task_skill_dataset(tasks, skills, candidates, evidence, graph, pilot, root / "out")
            self.assertEqual(report["api_calls_made"], 0)
            self.assertEqual(report["pair_count"], 20)
            self.assertEqual(report["task_counts_by_dataset_split"], {"dev": 1, "test": 1, "train": 8})
            pairs = [json.loads(line) for line in (root / "out/datasets/task_skill_v1/pairs.jsonl").read_text().splitlines()]
            negative = next(row for row in pairs if row["task_record_id"] == "t0" and row["skill_id"] == "s2")
            self.assertTrue(negative["is_hard_negative"])
            self.assertIn("graph_similar_hard_negative", negative["negative_types"])
            split_rows = [json.loads(line) for line in (root / "out/datasets/task_skill_v1/tasks.jsonl").read_text().splitlines()]
            self.assertEqual(next(row for row in split_rows if row["record_id"] == "t0")["dataset_split"], "train")


if __name__ == "__main__":
    unittest.main()
