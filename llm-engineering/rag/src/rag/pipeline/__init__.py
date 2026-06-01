from rag.pipeline.base import RAGPipeline
from rag.pipeline.vectordb import VectorDBPipeline
from rag.pipeline.conversational import ConversationalRAGPipeline
from rag.pipeline.factory import build_pipeline, build_hybrid_pipeline


def __getattr__(name):
    if name == "ChromaRAGPipeline":
        from rag.pipeline.chromadb import ChromaRAGPipeline
        return ChromaRAGPipeline
    if name == "QdrantRAGPipeline":
        from rag.pipeline.qdrant import QdrantRAGPipeline
        return QdrantRAGPipeline
    if name == "HybridRAGPipeline":
        from rag.pipeline.hybrid import HybridRAGPipeline
        return HybridRAGPipeline
    if name == "PostgresRAGPipeline":
        from rag.pipeline.postgres import PostgresRAGPipeline
        return PostgresRAGPipeline
    raise AttributeError(f"module 'rag.pipeline' has no attribute {name!r}")


__all__ = [
    "RAGPipeline",
    "VectorDBPipeline",
    "ConversationalRAGPipeline",
    "build_pipeline",
    "build_hybrid_pipeline",
    "ChromaRAGPipeline",
    "QdrantRAGPipeline",
    "HybridRAGPipeline",
    "PostgresRAGPipeline",
]
