from rag.databases.base import Database
from rag.pipeline.base import RAGPipeline
from rag.generators.base import Generator
from rag.chunker import Chunker, ParentChildChunker


class VectorDBPipeline(RAGPipeline):
    """RAG pipeline backed by a managed-embedding vector database (ChromaDB, Qdrant, etc.).

    The database handles embedding internally — just pass text in, get ranked results back.
    Use RAGPipeline directly when you want to control the embedder yourself.

    Usage:
        pipeline = VectorDBPipeline(db=ChromaDB("my_collection"))
        pipeline = VectorDBPipeline(db=QdrantDB("my_collection"))
    """

    def __init__(
        self,
        db: Database,
        generator: Generator | None = None,
        chunker: Chunker | ParentChildChunker | None = None,
        *,
        top_k: int = 5,
        chunk_size: int = 512,
        overlap: int = 50,
    ):
        super().__init__(generator=generator, chunker=chunker, top_k=top_k,
                         chunk_size=chunk_size, overlap=overlap)
        self.db = db

    def index(self, documents, source_names=None):
        all_chunks, sources, metadatas = self._prepare_chunks_and_sources(documents, source_names)
        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        self.db.add(all_chunks, ids, metadatas=metadatas)
        return len(all_chunks)

    def _retrieve(self, question, top_k):
        return [
            {
                "chunk": (r.get("metadata") or {}).get("parent_chunk", r["document"]),
                "id": r["id"],
                "score": r["score"],
                "source": (r.get("metadata") or {}).get("source", "unknown"),
                "chunk_position": (r.get("metadata") or {}).get("chunk_position", "unknown"),
            }
            for r in self.db.search(question, top_k)
        ]
