import unittest
from rag.evaluation import evaluate_faithfulness, evaluate_retrieval_recall


class TestEvaluationMetrics(unittest.TestCase):

    def test_faithfulness_grounded(self):
        chunks = [
            "Acme Corp standard plans are eligible for standard refunds within 30 days of purchase.",
            "Enterprise customers receive an extended 60-day refund window.",
        ]
        score, ungrounded = evaluate_faithfulness(
            "Standard plans get refunds within 30 days. Enterprise plan customers get 60 days.", chunks
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(len(ungrounded), 0)

    def test_faithfulness_hallucinated(self):
        chunks = [
            "Acme Corp standard plans are eligible for standard refunds within 30 days of purchase.",
            "Enterprise customers receive an extended 60-day refund window.",
        ]
        score, ungrounded = evaluate_faithfulness(
            "Standard plans get refunds within 30 days. Enterprise plans get a 90-day refund window with full cash back anytime.", chunks
        )
        self.assertLess(score, 1.0)
        self.assertEqual(len(ungrounded), 1)
        self.assertIn("90-day", ungrounded[0])

    def test_retrieval_recall_calculation(self):
        def mock_retrieval_fn(query, k):
            if "special" in query.lower():
                return [(0, 0.9), (1, 0.8)][:k]
            return [(0, 0.9)][:k]

        avg_recall, results = evaluate_retrieval_recall(
            [("Standard refund policies", [0, 1]), ("Special enterprise plans", [0])],
            mock_retrieval_fn,
            k=5,
        )
        self.assertEqual(avg_recall, 0.75)
        self.assertEqual(results[0]["recall"], 0.5)
        self.assertEqual(results[1]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
