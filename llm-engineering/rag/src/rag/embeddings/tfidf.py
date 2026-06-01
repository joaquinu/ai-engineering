import math
from collections import Counter
from rag.embeddings.base import Embeder


class TFIDFEmbeder(Embeder):
    def __init__(self, chunker=None):
        self.chunker = chunker
        self.vocabulary: list[str] = []
        self.idf: list[float] = []

    def build_vocabulary(self, documents: list[str]) -> None:
        vocab: set[str] = set()
        for doc in documents:
            words = [w.lower() for w in doc] if isinstance(doc, list) else doc.lower().split()
            vocab.update(words)
        self.vocabulary = sorted(vocab)

    def compute_idf(self, documents: list[str]) -> None:
        n = len(documents)
        self.idf = [
            math.log((n + 1) / (sum(
                1 for doc in documents
                if word in ([w.lower() for w in doc] if isinstance(doc, list) else doc.lower().split())
            ) + 1)) + 1
            for word in self.vocabulary
        ]

    def compute_tf(self, text: str, vocab: list[str] | None = None) -> list[float]:
        v = vocab if vocab is not None else self.vocabulary
        words = [w.lower() for w in text] if isinstance(text, list) else text.lower().split()
        count = Counter(words)
        total = len(words) if words else 1
        return [count.get(word, 0) / total for word in v]

    def tfidf_embed(self, text: str, vocab: list[str] | None = None, idf: list[float] | None = None) -> list[float]:
        v = vocab if vocab is not None else self.vocabulary
        i = idf if idf is not None else self.idf
        tf = self.compute_tf(text, v)
        return [t * val for t, val in zip(tf, i)]

    def embed(self, text: str) -> list[float]:
        return self.tfidf_embed(text)
