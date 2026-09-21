import json
import tempfile
import unittest
from pathlib import Path

from skilldag.data_pipeline import extract_static_data


class StaticDataExtractTests(unittest.TestCase):
    def test_extracts_versioned_tasks_skills_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alfworld = root / "alfworld"
            trajectory = alfworld / "train" / "task-a" / "traj_data.json"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_text(
                json.dumps(
                    {
                        "task_id": "task-a",
                        "task_type": "pick_and_place_simple",
                        "pddl_params": {"object_target": "Apple"},
                        "turk_annotations": {
                            "anns": [
                                {
                                    "task_desc": "Put the apple on the table.",
                                    "high_descs": ["Pick up the apple.", "Put it on the table."],
                                    "votes": [1, 1],
                                }
                            ]
                        },
                        "plan": {
                            "high_pddl": [
                                {"discrete_action": {"action": "PickupObject", "args": ["apple"]}}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            skills = root / "skills"
            skill_file = skills / "object-picker" / "SKILL.md"
            skill_file.parent.mkdir(parents=True)
            skill_file.write_text(
                "---\nname: object-picker\ndescription: Picks up an object.\n---\n# Picker\n",
                encoding="utf-8",
            )
            graph_path = root / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "schema_version": "skillgraph.v1",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "nodes": {
                            "object-picker": {
                                "name": "object-picker",
                                "description": "Picks up an object.",
                                "status": "active",
                                "tags": [],
                            }
                        },
                        "edges": [],
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )

            output = root / "output"
            report = extract_static_data(alfworld, skills, graph_path, output)

            self.assertEqual(report["tasks"]["count"], 1)
            self.assertEqual(report["skills"]["count"], 1)
            self.assertEqual(report["api_calls_made"], 0)
            task = json.loads((output / "raw" / "tasks.jsonl").read_text().strip())
            skill = json.loads((output / "raw" / "skills.jsonl").read_text().strip())
            self.assertEqual(task["task_text"], "Put the apple on the table.")
            self.assertEqual(task["plan_high_pddl"][0]["action"], "PickupObject")
            self.assertEqual(skill["skill_id"], "object-picker")
            self.assertEqual(skill["description"], "Picks up an object.")
            self.assertEqual(skill["graph_snapshot_id"], report["graph"]["graph_snapshot_id"])


if __name__ == "__main__":
    unittest.main()
