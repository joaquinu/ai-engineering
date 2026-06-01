from rag.pipeline.base import RAGPipeline


def __getattr__(name):
    _pipeline_names = {
        "ChromaRAGPipeline", "HybridRAGPipeline", "QdrantRAGPipeline",
        "PostgresRAGPipeline", "ConversationalRAGPipeline",
    }
    if name in _pipeline_names:
        import importlib
        mod = importlib.import_module("rag.pipeline")
        return getattr(mod, name)
    raise AttributeError(f"module 'rag' has no attribute {name!r}")


__all__ = [
    "RAGPipeline",
    "ChromaRAGPipeline",
    "HybridRAGPipeline",
    "QdrantRAGPipeline",
    "PostgresRAGPipeline",
    "ConversationalRAGPipeline",
]
