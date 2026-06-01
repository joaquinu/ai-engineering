from rag.embeddings.base import Embeder
from rag.retrieval import cosine_similarity


class SentenceTransformerEmbeder(Embeder):
    def __init__(self, chunker=None, model_name: str = "all-MiniLM-L6-v2"):
        self.chunker = chunker
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.vocabulary: list[str] = []
        self.idf: list[float] = []

    def embed(self, text: str) -> list[float]:
        if isinstance(text, list):
            text = " ".join(text)
        embedding = self.model.encode(text)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        return cosine_similarity(emb1, emb2)
