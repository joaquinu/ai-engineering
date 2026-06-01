import unittest
from unittest.mock import MagicMock, patch
from rag.reranker import Reranker
from rag.reranker.cohere import CohereReranker


class TestRerankers(unittest.TestCase):

    def test_simple_reranker_scoring(self):
        reranker = Reranker()
        chunks = [
            "This document describes the Acme Corp Refund Policy for Enterprise customers.",
            "This is the standard plan refund policy. No extra features.",
            "Machine learning and artificial intelligence are complex fields.",
        ]
        results = reranker.rerank("refund policy enterprise", [(0, 0.5), (1, 0.5), (2, 0.5)], chunks)
        self.assertEqual(results[0][0], 0)
        self.assertEqual(results[1][0], 1)
        self.assertEqual(results[2][0], 2)
        self.assertGreater(results[0][1], results[1][1])
        self.assertGreater(results[1][1], results[2][1])

    @patch("cohere.ClientV2")
    def test_cohere_reranker(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        r0 = MagicMock(score=0.95, index=0)
        r1 = MagicMock(score=0.65, index=1)
        mock_client.rerank.return_value = [r0, r1]

        reranker = CohereReranker()
        results = reranker.rerank(
            query="refund policy",
            documents=["Enterprise refund document", "Standard refund document"],
            top_k=2,
        )
        mock_client.rerank.assert_called_once()
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
