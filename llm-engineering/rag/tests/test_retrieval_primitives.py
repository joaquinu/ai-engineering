import unittest
from rag.retrieval import cosine_similarity, search, reciprocal_rank_fusion


class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors_return_one(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=6)

    def test_orthogonal_vectors_return_zero(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_opposite_vectors_return_negative_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, places=6)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_magnitude_invariance(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 1.0], [3.0, 3.0]), 1.0, places=6)

    def test_known_value(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0, 1.0], [1.0, 1.0, 0.0]), 0.5, places=6)


class TestSearch(unittest.TestCase):

    def setUp(self):
        self.embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        self.query = [1.0, 0.0]

    def test_top_k_returns_correct_count(self):
        self.assertEqual(len(search(self.query, self.embeddings, top_k=2)), 2)

    def test_results_ranked_by_decreasing_score(self):
        scores = [s for _, s in search(self.query, self.embeddings, top_k=3)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_best_match_is_first(self):
        self.assertEqual(search(self.query, self.embeddings, top_k=3)[0][0], 0)

    def test_top_k_larger_than_corpus(self):
        self.assertEqual(len(search(self.query, self.embeddings, top_k=100)), 3)

    def test_single_embedding(self):
        results = search([1.0, 0.0], [[1.0, 0.0]], top_k=1)
        self.assertAlmostEqual(results[0][1], 1.0, places=6)


class TestReciprocalRankFusion(unittest.TestCase):

    def test_document_appearing_in_both_lists_scores_higher(self):
        fused = reciprocal_rank_fusion([[(0, 0.9), (1, 0.7)], [(0, 0.8), (2, 0.6)]], k=60)
        self.assertEqual(fused[0][0], 0)

    def test_rrf_scores_decrease_monotonically(self):
        scores = [s for _, s in reciprocal_rank_fusion([[(0, 0.9), (1, 0.8), (2, 0.7)], [(0, 0.9), (1, 0.7), (2, 0.5)]], k=60)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_single_list_passthrough(self):
        doc_ids = {d for d, _ in reciprocal_rank_fusion([[(0, 0.9), (1, 0.5), (2, 0.1)]], k=60)}
        self.assertEqual(doc_ids, {0, 1, 2})

    def test_known_rrf_score(self):
        fused = reciprocal_rank_fusion([[(0, 0.0)]], k=60)
        self.assertAlmostEqual(fused[0][1], 1.0 / 61, places=8)

    def test_empty_lists_return_empty(self):
        self.assertEqual(reciprocal_rank_fusion([], k=60), [])

    def test_empty_ranked_list(self):
        self.assertEqual(reciprocal_rank_fusion([[]], k=60), [])


if __name__ == "__main__":
    unittest.main()
