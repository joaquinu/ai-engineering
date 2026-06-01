import unittest
from unittest.mock import MagicMock
from rag.generators.simple import SimpleGenerator
from rag.pipeline import RAGPipeline, HybridRAGPipeline

SAMPLE_DOCS = [
    "The Starter plan costs $29 per month. It includes 5 users and 10 GB of storage.",
    "The Professional plan costs $99 per month. It includes 25 users and 100 GB of storage.",
    "Enterprise customers receive a dedicated account manager and custom SLA agreements.",
    "Refund requests must be submitted within 30 days of purchase for all plans.",
    "Security features include AES-256 encryption and SOC2 Type II compliance.",
]


class TestSimpleGenerator(unittest.TestCase):

    def setUp(self):
        self.gen = SimpleGenerator()

    def test_returns_string(self):
        self.assertIsInstance(self.gen.generate("Question: What is the cost?", ["The Starter plan costs $29."]), str)

    def test_extracts_relevant_sentence(self):
        self.assertIn("$29", self.gen.generate("Question: cost starter plan", ["The Starter plan costs $29 per month. Other info here."]))

    def test_empty_chunks_returns_fallback(self):
        self.assertIn("I don't have enough information", self.gen.generate("Question: anything?", []))


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = RAGPipeline(chunk_size=30, overlap=5, top_k=3, generator_type="simple", embedder_type="tfidf")
        self.pipeline.index(SAMPLE_DOCS)

    def test_index_returns_positive_chunk_count(self):
        self.assertGreater(RAGPipeline(chunk_size=30, overlap=5).index(SAMPLE_DOCS), 0)

    def test_retrieve_returns_correct_number_of_results(self):
        self.assertEqual(len(self.pipeline._retrieve("refund policy", top_k=2)), 2)

    def test_retrieve_results_have_required_keys(self):
        r = self.pipeline._retrieve("starter plan cost", top_k=1)[0]
        for key in ("chunk", "score", "source", "chunk_position"):
            self.assertIn(key, r)

    def test_relevant_chunk_is_retrieved(self):
        combined = " ".join(r["chunk"] for r in self.pipeline._retrieve("refund request 30 days", top_k=3)).lower()
        self.assertIn("refund", combined)

    def test_query_returns_expected_structure(self):
        result = self.pipeline.query("How much does the Starter plan cost?")
        for key in ("answer", "retrieved", "prompt"):
            self.assertIn(key, result)

    def test_bow_embedder_type(self):
        p = RAGPipeline(chunk_size=30, overlap=5, embedder_type="bow")
        p.index(SAMPLE_DOCS)
        self.assertEqual(len(p._retrieve("refund", top_k=1)), 1)

    def test_bm25_embedder_type(self):
        p = RAGPipeline(chunk_size=30, overlap=5, embedder_type="bm25")
        p.index(SAMPLE_DOCS)
        self.assertEqual(len(p._retrieve("encryption security", top_k=2)), 2)

    def test_custom_source_names(self):
        p = RAGPipeline(chunk_size=50, overlap=0)
        sources = [f"custom_{i}.md" for i in range(len(SAMPLE_DOCS))]
        p.index(SAMPLE_DOCS, source_names=sources)
        self.assertIn("custom_", p._retrieve("starter plan", top_k=1)[0]["source"])


class TestHybridRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = HybridRAGPipeline(
            chunk_size=30, overlap=5, top_k=3, generator_type="simple",
            dense_embedder_type="bow", sparse_embedder_type="bow",
            use_reranker=False, use_hyde=False, verbose=False,
        )
        self.pipeline.index(SAMPLE_DOCS)

    def test_index_returns_positive_chunk_count(self):
        self.assertGreater(HybridRAGPipeline(dense_embedder_type="bow", sparse_embedder_type="bow", use_reranker=False, use_hyde=False, verbose=False).index(SAMPLE_DOCS), 0)

    def test_retrieve_returns_results(self):
        self.assertGreater(len(self.pipeline._retrieve("refund", top_k=2)), 0)

    def test_reranker_called_when_enabled(self):
        p = HybridRAGPipeline(chunk_size=30, overlap=5, top_k=3, dense_embedder_type="bow", sparse_embedder_type="bow", use_reranker=True, use_hyde=False, verbose=False)
        p.index(SAMPLE_DOCS)
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(0, 1.0)]
        p.reranker = mock_reranker
        p._retrieve("refund", top_k=1)
        mock_reranker.rerank.assert_called_once()

    def test_query_returns_expected_structure(self):
        result = self.pipeline.query("How much is the starter plan?")
        for key in ("answer", "retrieved", "prompt"):
            self.assertIn(key, result)


class TestConversationalRAGPipeline(unittest.TestCase):

    def setUp(self):
        from rag.pipeline import ConversationalRAGPipeline
        self.pipeline = ConversationalRAGPipeline(chunk_size=30, overlap=5, top_k=2, collection_name="test_conv_pipeline")
        self.pipeline.index(SAMPLE_DOCS)

    def test_first_query_returns_answer(self):
        result = self.pipeline.query("How much does the Starter plan cost?")
        self.assertIn("answer", result)
        self.assertIsInstance(result["answer"], str)

    def test_history_grows_after_query(self):
        self.pipeline.clear_history()
        self.pipeline.query("What is the refund policy?")
        self.assertEqual(len(self.pipeline.history), 1)

    def test_history_capped_at_three(self):
        self.pipeline.clear_history()
        for i in range(5):
            self.pipeline.query(f"Question number {i}?")
        self.assertLessEqual(len(self.pipeline.history), 3)

    def test_clear_history_resets_state(self):
        self.pipeline.query("Any question?")
        self.pipeline.clear_history()
        self.assertEqual(self.pipeline.history, [])

    def test_reformulate_query_no_history(self):
        self.pipeline.clear_history()
        self.assertEqual(self.pipeline.reformulate_query("standalone question?"), "standalone question?")

    def test_reformulate_short_followup_appends_root(self):
        self.pipeline.clear_history()
        self.pipeline.history = [{"user": "Tell me about pricing.", "assistant": "We have three plans."}]
        self.assertIn("Tell me about pricing.", self.pipeline.reformulate_query("what about enterprise?"))


if __name__ == "__main__":
    unittest.main()
