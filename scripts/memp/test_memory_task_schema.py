#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_task_schema import parse_query, relation_grade, script_validity
from calibrate_memory_task_labels import calibrated_grade, deduplicate_grade1_roles
from run_memory_task_judge import parse_response, prompt_for


class MemoryTaskSchemaTest(unittest.TestCase):
    def test_paraphrases_share_signature(self):
        self.assertEqual(
            parse_query("put a hot apple in fridge"),
            parse_query("heat some apple and put it in fridge"),
        )
        self.assertEqual(
            parse_query("put two cd in safe"),
            parse_query("find two cd and put them in safe"),
        )

    def test_graded_relations(self):
        task = parse_query("heat some apple and put it in garbagecan")
        self.assertEqual(relation_grade(task, parse_query("put a hot apple in garbagecan"))[0], 2)
        self.assertEqual(relation_grade(task, parse_query("put a hot apple in fridge"))[0], 1)
        self.assertEqual(relation_grade(task, parse_query("put a hot potato in garbagecan"))[0], 1)
        self.assertEqual(relation_grade(task, parse_query("put a cool apple in garbagecan"))[0], 0)

    def test_single_object_is_partial_for_pick_two(self):
        task = parse_query("find two cd and put them in safe")
        memory = parse_query("put a cd in safe")
        self.assertEqual(relation_grade(task, memory)[0], 1)

    def test_failure_targeted_relations_remain_conservative(self):
        task = parse_query("find two keychain and put them in safe")
        self.assertEqual(
            relation_grade(task, parse_query("put a keychain in safe")),
            (1, "single_object_subprocedure_for_pick_two"),
        )
        self.assertEqual(
            relation_grade(task, parse_query("put a book in safe")),
            (0, "same_destination_wrong_relation"),
        )
        self.assertEqual(
            relation_grade(task, parse_query("put two cd in drawer")),
            (0, "same_operation_unmatched_arguments"),
        )

    def test_invalid_pick_two_script_cannot_be_grade_two(self):
        signature = parse_query("find two cd and put them in safe")
        valid, _ = script_validity(
            signature,
            "Search until all required objects are in your possession, then place them one at a time.",
        )
        self.assertFalse(valid)
        self.assertEqual(relation_grade(signature, signature, memory_is_valid=valid)[0], 0)

    def test_judge_prompt_hides_rule_seed(self):
        batch = {
            "task_query": "put a hot apple in fridge",
            "task_signature": parse_query("put a hot apple in fridge").as_dict(),
            "candidates": [{
                "memory_id": 7,
                "memory_query": "heat some apple and put it in fridge",
                "workflow": "Heat the apple, then place it in the fridge.",
                "rule_seed_grade": 2,
                "rule_seed_reason": "exact_executable_match",
                "procedurally_valid_rule": True,
                "validity_reason_rule": "no_known_violation",
            }],
        }
        prompt = prompt_for(batch)
        self.assertIn("memory_id", prompt)
        self.assertNotIn("rule_seed", prompt)
        self.assertNotIn("procedurally_valid_rule", prompt)

    def test_query_only_judge_prompt_excludes_workflow(self):
        batch = {
            "task_query": "put a hot apple in fridge",
            "task_signature": parse_query("put a hot apple in fridge").as_dict(),
            "candidates": [{
                "memory_id": 7,
                "memory_query": "heat some apple and put it in fridge",
                "workflow": "PRIVATE WORKFLOW MUST NOT BE VISIBLE",
                "rule_seed_grade": 2,
            }],
        }
        prompt = prompt_for(batch, evidence_mode="query_only")
        self.assertIn("heat some apple", prompt)
        self.assertNotIn("PRIVATE WORKFLOW", prompt)
        self.assertNotIn("rule_seed_grade", prompt)

    def test_judge_response_requires_structured_evidence(self):
        response = '{"judgments":[{"memory_id":7,"grade":2,"confidence":0.91,"operation":"match","object":"match","cardinality":"match","destination":"match","procedural_validity":"valid","distinct_contribution":"complete procedure","reason":"explicit match"}]}'
        rows = parse_response(response, {7})
        self.assertEqual(rows[0]["grade"], 2)

    def test_judge_response_rejects_free_text_relation_fields(self):
        response = '{"judgments":[{"memory_id":7,"grade":2,"confidence":0.91,"operation":"heat","object":"apple","cardinality":"1","destination":"fridge","procedural_validity":"valid","distinct_contribution":"complete procedure","reason":"explicit match"}]}'
        with self.assertRaisesRegex(ValueError, "invalid operation enum"):
            parse_response(response, {7})

    def test_calibration_matches_skill_pipeline_principle(self):
        exact = {
            "rule_seed_grade": 2,
            "rule_seed_reason": "exact_executable_match",
            "procedurally_valid_rule": True,
        }
        conflict = {
            "rule_seed_grade": 0,
            "rule_seed_reason": "operation_or_cardinality_conflict",
            "procedurally_valid_rule": True,
        }
        invalid = {
            "rule_seed_grade": 2,
            "rule_seed_reason": "exact_executable_match",
            "procedurally_valid_rule": False,
        }
        self.assertEqual(calibrated_grade(1, exact)[0], 2)
        self.assertEqual(calibrated_grade(2, conflict)[0], 0)
        self.assertEqual(calibrated_grade(2, invalid)[0], 0)
        partial = {
            "rule_seed_grade": 1,
            "rule_seed_reason": "same_operation_object_different_destination",
            "procedurally_valid_rule": True,
        }
        self.assertEqual(calibrated_grade(2, partial)[0], 1)

    def test_rule_zero_cannot_be_promoted_by_judge(self):
        generic_template = {
            "rule_seed_grade": 0,
            "rule_seed_reason": "same_operation_unmatched_arguments",
            "procedurally_valid_rule": True,
        }
        self.assertEqual(calibrated_grade(1, generic_template)[0], 0)

    def test_grade_one_support_roles_are_deduplicated(self):
        rows = [
            {
                "task_id": 1,
                "memory_id": memory_id,
                "target_grade": 1,
                "target_relevance": 0.5,
                "label_confidence": confidence,
                "rule_seed_reason": "same_operation_object_different_destination",
                "calibration_action": "unchanged",
            }
            for memory_id, confidence in ((9, 0.80), (4, 0.90), (3, 0.90))
        ]
        deduplicate_grade1_roles(rows)
        kept = [row["memory_id"] for row in rows if row["target_grade"] == 1]
        self.assertEqual(kept, [3])
        self.assertEqual(sum(row["target_grade"] == 0 for row in rows), 2)


if __name__ == "__main__":
    unittest.main()
