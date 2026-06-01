from rag.reranker.base import Reranker


def __getattr__(name):
    if name == "CohereReranker":
        from rag.reranker.cohere import CohereReranker
        return CohereReranker
    raise AttributeError(f"module 'rag.reranker' has no attribute {name!r}")


__all__ = ["Reranker", "CohereReranker"]
