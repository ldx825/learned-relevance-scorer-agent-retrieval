import unittest

from benchmarks.alfworld.phase_context import build_phase_specs, build_task_structure
from benchmarks.alfworld.phase_online import (
    _execution_contract,
    render_phase_recommendations,
)


class PhaseOnlineContractTests(unittest.TestCase):
    def test_cool_contract_requires_explicit_environment_action(self):
        structure = build_task_structure(
            "pick_cool_then_place_in_recep",
            {
                "object_target": "Apple",
                "parent_target": "CounterTop",
                "object_sliced": False,
            },
        )
        transform = next(
            phase
            for phase in build_phase_specs(structure)
            if phase.phase_name == "transform_object"
        )
        contract = _execution_contract(structure, transform)
        self.assertEqual(contract["action_template"], "cool <object> with <fridge>")
        self.assertIn("does not cool", contract["non_example"])

    def test_render_places_contract_next_to_phase_recommendations(self):
        rendered = render_phase_recommendations(
            [
                {
                    "phase": "transform_object",
                    "query": "clean apple",
                    "execution_contract": {
                        "precondition": "hold the target object and stand at a sinkbasin",
                        "action_template": "clean <object> with <sinkbasin>",
                        "success_evidence": "the observation confirms cleaning",
                        "non_example": "moving into a sinkbasin does not clean it",
                    },
                    "matches": [
                        {
                            "skill_id": "alfworld-clean-object",
                            "description": "Clean a held object.",
                        }
                    ],
                }
            ]
        )
        self.assertIn("REQUIRED environment action", rendered)
        self.assertIn("clean <object> with <sinkbasin>", rendered)
        self.assertIn("alfworld-clean-object", rendered)


if __name__ == "__main__":
    unittest.main()
