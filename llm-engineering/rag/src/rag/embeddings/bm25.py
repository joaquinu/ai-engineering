import math
from collections import Counter
from rag.embeddings.base import Embeder


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[str] = []
        self.doc_lengths: list[int] = []
        self.avg_dl: float = 0
        self.doc_freqs: dict[str, int] = {}
        self.n_docs: int = 0

    def index(self, documents: list[str]) -> None:
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}
        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            for word in set(words):
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1
        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query: str, doc_idx: int) -> float:
        query_words = query.lower().split()
        word_counts = Counter(self.docs[doc_idx].lower().split())
        doc_len = self.doc_lengths[doc_idx]
        score = 0.0
        for term in query_words:
            tf = word_counts.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class BM25Embeder(Embeder):
    def __init__(self, chunker=None, k1: float = 1.2, b: float = 0.75):
        self.chunker = chunker
        self.bm25 = BM25(k1=k1, b=b)
        self.vocabulary: list[str] = []
        self.idf: list[float] = []

    def build_vocabulary(self, documents: list[str]) -> None:
        self.bm25.index(documents)
        self.vocabulary = sorted(self.bm25.doc_freqs.keys())
        self.idf = [
            max(math.log((self.bm25.n_docs - self.bm25.doc_freqs.get(w, 0) + 0.5) / (self.bm25.doc_freqs.get(w, 0) + 0.5) + 1), 0.0001)
            for w in self.vocabulary
        ]

    def compute_idf(self, documents: list[str]) -> None:
        pass  # handled in build_vocabulary via BM25.index

    def tfidf_embed(self, text: str, vocab=None, idf=None) -> list[float]:
        return self.embed(text)

    def embed(self, text: str) -> list[float]:
        words = text.lower().split() if isinstance(text, str) else [w.lower() for w in text]
        count = Counter(words)
        doc_len = len(words)

        if not self.bm25.docs or text not in self.bm25.docs:
            return [1.0 if word in words else 0.0 for word in self.vocabulary]

        weights = []
        for idx, word in enumerate(self.vocabulary):
            tf = count.get(word, 0)
            if tf == 0:
                weights.append(0.0)
                continue
            numerator = tf * (self.bm25.k1 + 1)
            denominator = tf + self.bm25.k1 * (1 - self.bm25.b + self.bm25.b * doc_len / self.bm25.avg_dl)
            weights.append(self.idf[idx] * numerator / denominator)
        return weights
