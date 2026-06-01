class Reranker:
    def __init__(self, stop_words=None):
        self.stop_words = stop_words if stop_words is not None else {
            "the", "a", "an", "is", "are", "was", "were", "what", "how",
            "why", "when", "where", "do", "does", "for", "of", "in", "to",
            "and", "or", "on", "at", "by", "it", "its", "this", "that",
            "with", "from", "be", "has", "have", "had", "not", "but",
        }

    def rerank(self, query: str, candidates: list[tuple[int, float]], chunks: list[str]) -> list[tuple[int, float]]:
        query_words = set(query.lower().split())
        query_terms = query_words - self.stop_words

        q_list = [w for w in query.lower().split() if w not in self.stop_words]
        query_bigrams = {q_list[i] + " " + q_list[i + 1] for i in range(len(q_list) - 1)}

        scored = []
        for doc_id, initial_score in candidates:
            chunk = chunks[doc_id].lower()
            chunk_words = set(chunk.split())
            term_overlap = len(query_terms & chunk_words)
            bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)
            position_boost = sum(
                0.5 for term in query_terms
                if (pos := chunk.find(term)) != -1 and pos < len(chunk) // 3
            )
            rerank_score = term_overlap * 1.0 + bigram_matches * 2.0 + position_boost + initial_score * 5.0
            scored.append((doc_id, rerank_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
