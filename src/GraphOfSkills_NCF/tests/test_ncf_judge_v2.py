import json
import tempfile
import unittest
from pathlib import Path

from gos.ncf.judge_v2 import _parse, run_gos_v2_judge, validate_task_evidence


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class JudgeV2Tests(unittest.TestCase):
    def test_task_evidence_must_be_exact_quote(self):
        self.assertTrue(validate_task_evidence("Cool a potato slice.", "cool a potato"))
        self.assertFalse(validate_task_evidence("Cool a potato slice.", "reduce temperature"))

    def test_invalid_primary_evidence_is_downgraded_to_uncertain(self):
        response = {
            "task_signature": {"core_objective": "cool potato"},
            "primary_skills": [
                {"skill_id": "cooler", "task_evidence": "reduce temperature"}
            ],
            "support_skill_ids": [],
            "harmful_skill_ids": [],
            "uncertain_skill_ids": [],
            "minimal_set_reason": "cooling",
        }
        parsed = _parse(json.dumps(response), {"cooler"}, "Cool a potato slice.")
        row = parsed["judgments"][0]
        self.assertEqual(row["role"], "uncertain")
        self.assertFalse(row["task_evidence_valid"])
        self.assertIn("invalid_primary_task_evidence", row["quality_flags"])

    def test_collection_is_cached_and_uses_soft_training_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, skills = root / "tasks.jsonl", root / "skills.jsonl"
            candidates, selection = root / "candidates.jsonl", root / "selection.json"
            write_jsonl(tasks, [{"record_id": "t", "task_text": "Cool a potato."}])
            write_jsonl(skills, [
                {"skill_id": "cooler", "description": "cools"},
                {"skill_id": "navigator", "description": "navigates"},
                {"skill_id": "heater", "description": "heats"},
            ])
            write_jsonl(candidates, [
                {"task_record_id": "t", "skill_id": "cooler"},
                {"task_record_id": "t", "skill_id": "navigator"},
                {"task_record_id": "t", "skill_id": "heater"},
            ])
            selection.write_text(json.dumps({"record_ids": ["t"]}))
            calls = []

            def chat(_payload):
                calls.append(1)
                return {"choices": [{"message": {"content": json.dumps({
                    "task_signature": {"core_objective": "cool"},
                    "primary_skills": [
                        {"skill_id": "cooler", "task_evidence": "Cool a potato"}
                    ],
                    "support_skill_ids": ["navigator"],
                    "harmful_skill_ids": [],
                    "uncertain_skill_ids": [],
                    "minimal_set_reason": "cool then continue",
                })}}]}

            first = run_gos_v2_judge(
                tasks, skills, candidates, selection, root / "out", "model", chat,
                collection_name="test", max_workers=1,
            )
            second = run_gos_v2_judge(
                tasks, skills, candidates, selection, root / "out", "model", chat,
                collection_name="test", max_workers=1,
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(second["cache_hit_task_count"], 1)
            labels = [
                json.loads(line)
                for line in Path(first["outputs"]["labels"]).read_text().splitlines()
            ]
            targets = {row["skill_id"]: row["target_score"] for row in labels}
            self.assertEqual(
                targets, {"cooler": 1.0, "navigator": 0.7, "heater": 0.0}
            )
            weights = {row["skill_id"]: row["sample_weight"] for row in labels}
            self.assertEqual(
                weights, {"cooler": 1.0, "navigator": 0.7, "heater": 0.7}
            )


if __name__ == "__main__":
    unittest.main()
