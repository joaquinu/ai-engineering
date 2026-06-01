from rag.embeddings.base import Embeder


class BinaryBOWEmbeder(Embeder):
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

    def embed(self, text: str) -> list[float]:
        words = {w.lower() for w in text} if isinstance(text, list) else {w.lower() for w in text.split()}
        return [1.0 if word in words else 0.0 for word in self.vocabulary]
