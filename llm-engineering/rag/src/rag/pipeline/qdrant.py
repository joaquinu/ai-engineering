from rag.databases.qdrant import QdrantDB
from rag.pipeline.vectordb import VectorDBPipeline
from rag.generators.base import Generator
from rag.chunker import Chunker, ParentChildChunker


class QdrantRAGPipeline(VectorDBPipeline):
    """VectorDBPipeline pre-configured with QdrantDB. Kept for backward compatibility."""

    def __init__(
        self,
        collection_name: str = "qdrant_collection",
        generator: Generator | None = None,
        chunker: Chunker | ParentChildChunker | None = None,
        *,
        top_k: int = 5,
        chunk_size: int = 512,
        overlap: int = 50,
    ):
        super().__init__(db=QdrantDB(collection=collection_name), generator=generator,
                         chunker=chunker, top_k=top_k, chunk_size=chunk_size, overlap=overlap)
