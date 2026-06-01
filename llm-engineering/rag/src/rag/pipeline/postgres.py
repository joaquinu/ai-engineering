from rag.databases.postgres import PostgresDB
from rag.pipeline.base import RAGPipeline
from rag.embeddings.base import Embeder
from rag.generators.base import Generator
from rag.chunker import Chunker, ParentChildChunker


class PostgresRAGPipeline(RAGPipeline):
    """RAGPipeline that persists embeddings in PostgreSQL via pgvector.

    Uses your own embedder (same as the base pipeline) but stores vectors in
    Postgres instead of memory, enabling persistence across process restarts.
    """

    def __init__(
        self,
        embedder: Embeder | None = None,
        generator: Generator | None = None,
        chunker: Chunker | ParentChildChunker | None = None,
        *,
        collection_name: str = "postgres_collection",
        top_k: int = 5,
        chunk_size: int = 512,
        overlap: int = 50,
        verbose: bool = True,
    ):
        super().__init__(embedder=embedder, generator=generator, chunker=chunker,
                         top_k=top_k, chunk_size=chunk_size, overlap=overlap)
        self.db = PostgresDB(table_name=collection_name)
        self.verbose = verbose

    def _store_index(self, embeddings: list) -> None:
        ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        self.db.add(self.chunks, ids, embeddings, metadatas=self.metadatas)

    def _retrieve(self, question, top_k):
        query_emb = self.embedder.embed(question)
        return [
            {
                "chunk": (r.get("metadata") or {}).get("parent_chunk", r["document"]),
                "id": r["id"],
                "score": r["score"],
                "source": (r.get("metadata") or {}).get("source", "unknown"),
                "chunk_position": (r.get("metadata") or {}).get("chunk_position", "unknown"),
            }
            for r in self.db.query(query_emb, n_results=top_k)
        ]
