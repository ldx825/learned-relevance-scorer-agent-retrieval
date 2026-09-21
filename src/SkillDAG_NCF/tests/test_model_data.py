import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401

    from skilldag.data_pipeline.model_baselines import evaluate_model_baselines
    from skilldag.data_pipeline.model_data import TaskSkillDataset, prepare_model_data

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False


@unittest.skipUnless(HAS_NUMPY, "requires the optional ncf dependency (NumPy)")
class ModelDataTests(unittest.TestCase):
    def test_prepares_student_features_batches_and_baselines_without_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            tasks = [
                {"record_id": "t-dev", "dataset_split": "dev"},
                {"record_id": "t-test", "dataset_split": "test"},
            ]
            skills = [{"skill_id": "s-good"}, {"skill_id": "s-bad"}]
            pairs = []
            for task in tasks:
                for rank, skill in enumerate(skills, 1):
                    good = skill["skill_id"] == "s-good"
                    pairs.append({
                        "pair_id": f"{task['record_id']}:{skill['skill_id']}",
                        "task_record_id": task["record_id"], "skill_id": skill["skill_id"],
                        "dataset_split": task["dataset_split"],
                        "target_grade": 2 if good else 0,
                        "target_relevance": 1.0 if good else 0.0,
                        "sample_weight": 0.5, "raw_label": "required" if good else "irrelevant",
                        "is_harmful": False, "task_semantic_rank": rank,
                        "action_semantic_rank": rank,
                        "graph_relations_to_candidates": [{
                            "skill_id": "s-bad" if good else "s-good",
                            "type": "similar_to", "direction": "outgoing" if good else "incoming",
                        }],
                    })
            for name, rows in (("tasks.jsonl", tasks), ("skills.jsonl", skills), ("pairs.jsonl", pairs)):
                (dataset / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
            (dataset / "splits.json").write_text(json.dumps({"record_ids": {}}))

            task_cache = root / "task_cache"
            task_cache.mkdir()
            (task_cache / "shard_00000.json").write_text(json.dumps({
                "t-dev": {"model": "m", "embedding": [1.0, 0.0]},
                "t-test": {"model": "m", "embedding": [1.0, 0.0]},
            }))
            skill_cache = root / "skills_cache.json"
            skill_cache.write_text(json.dumps({
                "s-good": {"model": "m", "embedding": [1.0, 0.0]},
                "s-bad": {"model": "m", "embedding": [0.0, 1.0]},
            }))
            graph = root / "graph.json"
            graph.write_text(json.dumps({"edges": [{
                "source": "s-good", "target": "s-bad", "type": "similar_to"
            }]}))

            metadata = prepare_model_data(dataset, task_cache, skill_cache, graph, root / "out")
            self.assertEqual(metadata["api_calls_made"], 0)
            self.assertNotIn("action_cosine_score", metadata["student_feature_contract"]["allowed"])
            arrays = root / "out/model_data/task_skill_v1/arrays.npz"
            dev = TaskSkillDataset(arrays, split="dev")
            self.assertEqual(len(dev), 2)
            batch = next(dev.iter_batches(2, shuffle=True, seed=1))
            self.assertEqual(batch["task_embeddings"].shape, (2, 2))
            report = evaluate_model_baselines(dataset, arrays, root / "out")
            self.assertEqual(report["trained_models"], 0)
            self.assertEqual(report["subsets"]["dev"]["methods"]["student_cosine"]["ndcg@3"], 1.0)


if __name__ == "__main__":
    unittest.main()
