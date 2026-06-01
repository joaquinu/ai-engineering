import unittest
from unittest.mock import MagicMock, patch
from rag.generators import ClaudeGenerator
from rag.pipeline import HybridRAGPipeline


class TestHyDE(unittest.TestCase):

    def test_hyde_offline_fallback(self):
        generator = ClaudeGenerator()
        generator.client = None
        hypothetical_doc = generator.hyde_with_llm("What is the refund policy for enterprise customers?")
        self.assertIn("Acme Corporation", hypothetical_doc)
        self.assertIn("refund", hypothetical_doc)

    @patch("anthropic.Anthropic")
    def test_hyde_with_mocked_llm(self, mock_anthropic_class):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="This is a mocked hypothetical refund policy response.")]
        )
        generator = ClaudeGenerator()
        self.assertEqual(generator.hyde_with_llm("What is the refund policy?"), "This is a mocked hypothetical refund policy response.")

    @patch("rag.generators.ClaudeGenerator")
    def test_hybrid_pipeline_uses_hyde(self, mock_generator_class):
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.hyde_with_llm.return_value = "Simulated hypothetical document"

        pipeline = HybridRAGPipeline(
            use_hyde=True, use_reranker=False,
            dense_embedder_type="bow", sparse_embedder_type="bow", verbose=False,
        )
        pipeline.sparse_embedder = MagicMock()
        pipeline.dense_embedder = MagicMock()
        pipeline.sparse_embeddings = [[0.0]]
        pipeline.dense_embeddings = [[0.0]]
        pipeline.chunks = ["Acme corp document."]
        pipeline.sources = ["doc.md"]
        pipeline.metadatas = [{}]

        pipeline._retrieve("What is the refund policy?", top_k=1)
        mock_generator.hyde_with_llm.assert_called_once_with("What is the refund policy?")


if __name__ == "__main__":
    unittest.main()
