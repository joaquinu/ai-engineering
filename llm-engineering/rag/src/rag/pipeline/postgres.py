from rag.databases.postgres import PostgresDB
from rag.pipeline.base import RAGPipeline


class PostgresRAGPipeline(RAGPipeline):
    def __init__(self, chunk_size=512, overlap=50, top_k=5, generator_type="simple",
                 collection_name="postgres_collection", chunker_type="standard",
                 embedder_type="sentence_transformers", verbose=True, **kwargs):
        super().__init__(chunk_size=chunk_size, overlap=overlap, top_k=top_k,
                         generator_type=generator_type, chunker_type=chunker_type,
                         embedder_type=embedder_type)
        self.db = PostgresDB(table_name=collection_name)
        self.verbose = verbose

    def index(self, documents, source_names=None):
        all_chunks, sources, metadatas = self._prepare_chunks_and_sources(documents, source_names)
        self.chunks = all_chunks
        self.sources = sources
        self.metadatas = metadatas
        self.embedder.build_vocabulary(all_chunks)
        if hasattr(self.embedder, "compute_idf"):
            self.embedder.compute_idf(all_chunks)
            self.idf = self.embedder.idf
        self.vocab = self.embedder.vocabulary
        self.embeddings = [self.embedder.embed(chunk) for chunk in self.chunks]
        ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        self.db.add(self.chunks, ids, self.embeddings, metadatas=self.metadatas)
        return len(self.chunks)

    def _retrieve(self, question, top_k):
        if hasattr(self.embedder, "tfidf_embed") and self.vocab and self.idf:
            query_emb = self.embedder.tfidf_embed(question, self.vocab, self.idf)
        else:
            query_emb = self.embedder.embed(question)

        retrieved_list = []
        for r in self.db.query(query_emb, n_results=top_k):
            meta = r.get("metadata") or {}
            retrieved_list.append({
                "chunk": meta.get("parent_chunk", r["document"]),
                "id": r["id"],
                "score": r["score"],
                "source": meta.get("source", "unknown"),
                "chunk_position": meta.get("chunk_position", "unknown"),
            })
        return retrieved_list
