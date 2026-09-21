import json
import tempfile
import unittest
from pathlib import Path

from gos.ncf.audit_v2 import requested_transformations
from gos.ncf.dataset_v2 import clean_gos_v2_labels


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class DatasetV2Tests(unittest.TestCase):
    def test_requested_operations_use_text_and_goal_type_without_fake_slice_skill(self):
        self.assertEqual(
            requested_transformations(
                {
                    "task_text": "Put a bowl with water in a cabinet.",
                    "task_type": "pick_clean_then_place_in_recep",
                }
            ),
            {"clean"},
        )
        self.assertEqual(
            requested_transformations(
                {
                    "task_text": "Slice a tomato on the table.",
                    "task_type": "pick_and_place_simple",
                }
            ),
            set(),
        )
        self.assertEqual(
            requested_transformations(
                {
                    "task_text": "Put a cold potato in the microwave.",
                    "task_type": "pick_cool_then_place_in_recep",
                }
            ),
            {"cool"},
        )

    def test_cleaning_repairs_roles_and_keeps_negative_weights_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            labels = root / "labels.jsonl"
            candidates = root / "candidates.jsonl"
            output = root / "clean.jsonl"
            report = root / "report.json"
            clean_id = "alfworld-clean-object"
            heat_id = "alfworld-object-heater"
            cool_id = "alfworld-object-cooler"
            write_jsonl(
                tasks,
                [
                    {
                        "record_id": "clean-task",
                        "task_text": "Put a washed bowl away.",
                        "task_type": "pick_clean_then_place_in_recep",
                    },
                    {
                        "record_id": "cool-task",
                        "task_text": "Put a cold potato in the microwave.",
                        "task_type": "pick_cool_then_place_in_recep",
                    },
                ],
            )
            write_jsonl(
                labels,
                [
                    {
                        "task_record_id": "clean-task",
                        "skill_id": clean_id,
                        "role": "harmful",
                        "target_grade": 0,
                        "target_score": 0.0,
                        "sample_weight": 0.0,
                        "quality_flags": [],
                    },
                    {
                        "task_record_id": "cool-task",
                        "skill_id": cool_id,
                        "role": "primary",
                        "target_grade": 2,
                        "target_score": 1.0,
                        "sample_weight": 1.0,
                        "quality_flags": [],
                    },
                    {
                        "task_record_id": "cool-task",
                        "skill_id": heat_id,
                        "role": "primary",
                        "target_grade": 2,
                        "target_score": 1.0,
                        "sample_weight": 1.0,
                        "quality_flags": [],
                    },
                ],
            )
            write_jsonl(
                candidates,
                [
                    {
                        "task_record_id": "clean-task",
                        "skill_id": clean_id,
                        "semantic_cosine": 0.4,
                    },
                    {
                        "task_record_id": "cool-task",
                        "skill_id": cool_id,
                        "semantic_cosine": 0.5,
                    },
                    {
                        "task_record_id": "cool-task",
                        "skill_id": heat_id,
                        "semantic_cosine": 0.2,
                    },
                ],
            )

            result = clean_gos_v2_labels(
                tasks, labels, candidates, output, report
            )
            rows = {
                (row["task_record_id"], row["skill_id"]): row
                for row in map(json.loads, output.read_text().splitlines())
            }
            repaired = rows[("clean-task", clean_id)]
            demoted = rows[("cool-task", heat_id)]
            self.assertEqual(repaired["role"], "primary")
            self.assertEqual(repaired["audit_original_role"], "harmful")
            self.assertEqual(demoted["role"], "irrelevant")
            self.assertEqual(demoted["sample_weight"], 0.7)
            self.assertEqual(result["change_types"], {"promoted": 1, "demoted": 1})


if __name__ == "__main__":
    unittest.main()
