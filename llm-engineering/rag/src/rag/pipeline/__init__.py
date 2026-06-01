from rag.pipeline.base import RAGPipeline


def __getattr__(name):
    if name == "ChromaRAGPipeline":
        from rag.pipeline.chromadb import ChromaRAGPipeline
        return ChromaRAGPipeline
    if name == "ConversationalRAGPipeline":
        from rag.pipeline.conversational import ConversationalRAGPipeline
        return ConversationalRAGPipeline
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
    "ChromaRAGPipeline",
    "ConversationalRAGPipeline",
    "QdrantRAGPipeline",
    "HybridRAGPipeline",
    "PostgresRAGPipeline",
]
