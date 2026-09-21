import unittest

import numpy as np

from evaluate_memory_retriever_table import metrics_for_rankings


class RetrievalTableMetricTest(unittest.TestCase):
    def test_metrics_use_grade2_for_recall_and_graded_ndcg(self):
        grades = np.asarray([[2, 1, 0], [0, 2, 1]], dtype=np.int8)
        rankings = np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
        result = metrics_for_rankings(grades, rankings, final_k=3)
        self.assertEqual(result["grade_2_eligible_query_count"], 2)
        self.assertEqual(result["ret@1"], 0.5)
        self.assertEqual(result["ret@5"], 1.0)
        self.assertAlmostEqual(result["mrr@10"], (1.0 + 1.0 / 3.0) / 2.0)
        self.assertAlmostEqual(result["compatible_precision@10"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
