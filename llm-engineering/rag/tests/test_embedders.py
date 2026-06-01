import unittest
from rag.chunker import Chunker
from rag.retrieval import cosine_similarity
from rag.embeddings import TFIDFEmbeder, BinaryBOWEmbeder
from rag.embeddings.bm25 import BM25, BM25Embeder

CORPUS = [
    "refund policy enterprise customers extended window",
    "standard plan refund thirty days purchase",
    "security encryption data protection privacy",
]


class TestTFIDFEmbeder(unittest.TestCase):

    def setUp(self):
        self.embedder = TFIDFEmbeder(Chunker(max_tokens=50, overlap=0))
        self.embedder.build_vocabulary(CORPUS)
        self.embedder.compute_idf(CORPUS)

    def test_vocabulary_is_sorted(self):
        self.assertEqual(self.embedder.vocabulary, sorted(self.embedder.vocabulary))

    def test_vocabulary_contains_corpus_words(self):
        for word in ["refund", "enterprise", "security"]:
            self.assertIn(word, self.embedder.vocabulary)

    def test_embed_length_matches_vocabulary(self):
        self.assertEqual(len(self.embedder.embed(CORPUS[0])), len(self.embedder.vocabulary))

    def test_embed_is_non_negative(self):
        self.assertTrue(all(v >= 0 for v in self.embedder.embed(CORPUS[0])))

    def test_similar_docs_closer_than_dissimilar(self):
        v0, v1, v2 = [self.embedder.embed(c) for c in CORPUS]
        self.assertGreater(cosine_similarity(v0, v1), cosine_similarity(v0, v2))

    def test_unknown_word_does_not_crash(self):
        self.assertEqual(len(self.embedder.embed("completely unknown zzz xyz")), len(self.embedder.vocabulary))

    def test_tfidf_embed_with_explicit_vocab_and_idf(self):
        vec = self.embedder.tfidf_embed(CORPUS[0], self.embedder.vocabulary, self.embedder.idf)
        self.assertEqual(len(vec), len(self.embedder.vocabulary))


class TestBinaryBOWEmbeder(unittest.TestCase):

    def setUp(self):
        self.embedder = BinaryBOWEmbeder(Chunker(max_tokens=50, overlap=0))
        self.embedder.build_vocabulary(CORPUS)

    def test_vocabulary_is_sorted(self):
        self.assertEqual(self.embedder.vocabulary, sorted(self.embedder.vocabulary))

    def test_embed_is_binary(self):
        for v in self.embedder.embed(CORPUS[0]):
            self.assertIn(v, (0.0, 1.0))

    def test_known_word_is_one(self):
        vec = self.embedder.embed("refund")
        self.assertEqual(vec[self.embedder.vocabulary.index("refund")], 1.0)

    def test_unknown_word_produces_all_zeros(self):
        self.assertTrue(all(v == 0.0 for v in self.embedder.embed("zzzzunknownword")))

    def test_list_input_tokenised_correctly(self):
        self.assertEqual(self.embedder.embed("refund security"), self.embedder.embed(["refund", "security"]))

    def test_embed_length_matches_vocabulary(self):
        self.assertEqual(len(self.embedder.embed(CORPUS[1])), len(self.embedder.vocabulary))


class TestBM25(unittest.TestCase):

    def setUp(self):
        self.bm25 = BM25(k1=1.2, b=0.75)
        self.bm25.index(CORPUS)

    def test_index_sets_n_docs(self):
        self.assertEqual(self.bm25.n_docs, len(CORPUS))

    def test_score_zero_for_unrelated_doc(self):
        self.assertEqual(self.bm25.score("security", 1), 0.0)

    def test_relevant_doc_scores_higher(self):
        self.assertGreater(self.bm25.score("enterprise", 0), self.bm25.score("enterprise", 2))

    def test_search_returns_sorted_results(self):
        scores = [s for _, s in self.bm25.search("refund", top_k=3)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_top_k_respected(self):
        self.assertEqual(len(self.bm25.search("refund", top_k=1)), 1)

    def test_search_top_k_larger_than_corpus(self):
        self.assertEqual(len(self.bm25.search("refund", top_k=100)), len(CORPUS))


class TestBM25Embeder(unittest.TestCase):

    def setUp(self):
        self.embedder = BM25Embeder(Chunker(max_tokens=50, overlap=0))
        self.embedder.build_vocabulary(CORPUS)

    def test_vocabulary_built(self):
        self.assertGreater(len(self.embedder.vocabulary), 0)

    def test_embed_length_matches_vocabulary(self):
        self.assertEqual(len(self.embedder.embed(CORPUS[0])), len(self.embedder.vocabulary))

    def test_query_embed_is_binary(self):
        for v in self.embedder.embed("enterprise refund"):
            self.assertIn(v, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
