from abc import ABC, abstractmethod


class Embeder(ABC):
    vocabulary: list[str] = []
    idf: list[float] = []

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def build_vocabulary(self, documents: list[str]) -> None:
        pass

    def compute_idf(self, documents: list[str]) -> None:
        pass

    def tfidf_embed(self, text: str, vocab=None, idf=None) -> list[float]:
        return self.embed(text)
