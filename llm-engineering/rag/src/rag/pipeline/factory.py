"""
String-based convenience constructors for RAGPipeline and HybridRAGPipeline.

Use these when pipeline configuration comes from config files, API requests,
or CLI flags. For code that lives next to the pipeline, prefer direct injection:

    RAGPipeline(embedder=BM25Embeder(), generator=ClaudeGenerator())
"""
from __future__ import annotations


def _make_embedder(name: str):
    name = name.lower()
    if name == "bow":
        from rag.embeddings import BinaryBOWEmbeder
        return BinaryBOWEmbeder()
    if name == "bm25":
        from rag.embeddings import BM25Embeder
        return BM25Embeder()
    if name in ("sentence_transformers", "st"):
        from rag.embeddings import SentenceTransformerEmbeder
        return SentenceTransformerEmbeder()
    from rag.embeddings import TFIDFEmbeder
    return TFIDFEmbeder()


def _make_generator(name: str):
    name = name.lower()
    if name == "claude":
        from rag.generators import ClaudeGenerator
        return ClaudeGenerator()
    if name == "openai":
        from rag.generators.openai import OpenAIGenerator
        return OpenAIGenerator()
    from rag.generators import SimpleGenerator
    return SimpleGenerator()


def _make_reranker(name: str):
    if name == "cohere":
        from rag.reranker import CohereReranker
        return CohereReranker()
    from rag.reranker import Reranker
    return Reranker()


def _make_chunker(chunker_type: str, chunk_size: int, overlap: int):
    if chunker_type == "parent_child":
        from rag.chunker import ParentChildChunker
        return ParentChildChunker(
            parent_size=chunk_size,
            parent_overlap=overlap,
            child_size=max(32, chunk_size // 4),
            child_overlap=max(5, overlap // 5),
        )
    from rag.chunker import Chunker
    return Chunker(chunk_size, overlap)


def build_pipeline(
    embedder: str = "tfidf",
    generator: str = "simple",
    chunker: str = "standard",
    chunk_size: int = 512,
    overlap: int = 50,
    top_k: int = 5,
) -> "RAGPipeline":
    """Build a RAGPipeline from string identifiers."""
    from rag.pipeline.base import RAGPipeline
    return RAGPipeline(
        embedder=_make_embedder(embedder),
        generator=_make_generator(generator),
        chunker=_make_chunker(chunker, chunk_size, overlap),
        top_k=top_k,
    )


def build_hybrid_pipeline(
    sparse_embedder: str = "bm25",
    dense_embedder: str = "sentence_transformers",
    generator: str = "simple",
    reranker: str = "simple",
    chunker: str = "standard",
    chunk_size: int = 512,
    overlap: int = 50,
    top_k: int = 5,
    rrf_k: int = 60,
    use_reranker: bool = True,
    use_hyde: bool = True,
    verbose: bool = True,
) -> "HybridRAGPipeline":
    """Build a HybridRAGPipeline from string identifiers."""
    from rag.pipeline.hybrid import HybridRAGPipeline
    return HybridRAGPipeline(
        sparse_embedder=_make_embedder(sparse_embedder),
        dense_embedder=_make_embedder(dense_embedder),
        generator=_make_generator(generator),
        reranker=_make_reranker(reranker),
        chunker=_make_chunker(chunker, chunk_size, overlap),
        top_k=top_k,
        rrf_k=rrf_k,
        use_reranker=use_reranker,
        use_hyde=use_hyde,
        verbose=verbose,
    )
