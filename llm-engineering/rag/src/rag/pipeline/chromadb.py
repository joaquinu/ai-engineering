from rag.databases.chromadb import ChromaDB
from rag.pipeline.base import RAGPipeline


class ChromaRAGPipeline(RAGPipeline):
    def __init__(self, chunk_size=512, overlap=50, top_k=5, generator_type="simple",
                 collection_name="rag_collection", chunker_type="standard"):
        super().__init__(chunk_size=chunk_size, overlap=overlap, top_k=top_k,
                         generator_type=generator_type, chunker_type=chunker_type)
        self.db = ChromaDB(collection=collection_name)

    def index(self, documents, source_names=None):
        all_chunks, sources, metadatas = self._prepare_chunks_and_sources(documents, source_names)
        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        self.db.add(all_chunks, ids, metadatas=metadatas)
        return len(all_chunks)

    def _retrieve(self, question, top_k):
        results = self.db.query(question, n_results=top_k)
        retrieved_list = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [None] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            for doc, c_id, meta, dist in zip(docs, ids, metadatas, distances):
                chunk_text = meta.get("parent_chunk", doc) if meta else doc
                retrieved_list.append({
                    "chunk": chunk_text,
                    "id": c_id,
                    "distance": dist,
                    "score": 1 - dist if dist <= 1.0 else 0.0,
                    "source": meta.get("source", "unknown") if meta else "unknown",
                    "chunk_position": meta.get("chunk_position", "unknown") if meta else "unknown",
                })
        return retrieved_list
