from rag.databases.qdrant import QdrantDB
from rag.pipeline.base import RAGPipeline


class QdrantRAGPipeline(RAGPipeline):
    def __init__(self, chunk_size=512, overlap=50, top_k=5, generator_type="simple",
                 collection_name="qdrant_collection", chunker_type="standard"):
        super().__init__(chunk_size=chunk_size, overlap=overlap, top_k=top_k,
                         generator_type=generator_type, chunker_type=chunker_type)
        self.db = QdrantDB(collection=collection_name)

    def index(self, documents, source_names=None):
        all_chunks, sources, metadatas = self._prepare_chunks_and_sources(documents, source_names)
        self.db.add(all_chunks, list(range(len(all_chunks))), metadatas=metadatas)
        return len(all_chunks)

    def _retrieve(self, question, top_k):
        retrieved_list = []
        for r in self.db.query(question, n_results=top_k):
            meta = r.metadata if r.metadata else {}
            retrieved_list.append({
                "chunk": meta.get("parent_chunk", r.document),
                "id": r.id,
                "score": r.score,
                "source": meta.get("source", "unknown"),
                "chunk_position": meta.get("chunk_position", "unknown"),
            })
        return retrieved_list
