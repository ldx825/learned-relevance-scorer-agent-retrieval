import json
import tempfile
import unittest
from pathlib import Path

from skilldag.data_pipeline import build_full_split_task_candidates


class FullTaskCandidateTests(unittest.TestCase):
    def test_sharded_cache_resumes_without_embedding_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks_path = root / "tasks.jsonl"
            tasks = [
                {
                    "record_id": f"task:{index}",
                    "source_task_id": str(index),
                    "split": "train",
                    "task_type": "type-a",
                    "task_text": f"task text {index}",
                }
                for index in range(3)
            ]
            tasks_path.write_text("".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8")
            action_path = root / "actions.jsonl"
            action_path.write_text("", encoding="utf-8")
            graph_path = root / "graph.json"
            graph_path.write_text(json.dumps({"nodes": {"skill-a": {}, "skill-b": {}}}), encoding="utf-8")
            skills_path = root / "skills.json"
            skills_path.write_text(
                json.dumps(
                    {
                        "skill-a": {"model": "test", "embedding": [1.0, 0.0]},
                        "skill-b": {"model": "test", "embedding": [0.0, 1.0]},
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def embed(texts):
                calls.append(list(texts))
                return [[1.0, 0.0] for _ in texts]

            output = root / "output"
            first = build_full_split_task_candidates(
                tasks_path,
                action_path,
                graph_path,
                skills_path,
                output,
                "test",
                embed,
                task_top_k=1,
                batch_size=2,
            )
            self.assertEqual(first["embedding_usage"]["task_vectors_computed"], 3)
            self.assertEqual(len(calls), 2)

            def no_api(_):
                raise AssertionError("cache should prevent another API call")

            second = build_full_split_task_candidates(
                tasks_path,
                action_path,
                graph_path,
                skills_path,
                output,
                "test",
                no_api,
                task_top_k=1,
                batch_size=2,
            )
            self.assertEqual(second["embedding_usage"]["task_vectors_computed"], 0)
            self.assertEqual(second["embedding_usage"]["task_vectors_reused_from_shards"], 3)
            self.assertEqual(second["candidates"]["task_semantic_pair_count"], 3)


if __name__ == "__main__":
    unittest.main()
