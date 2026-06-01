import concurrent.futures
from rag.pipeline.base import RAGPipeline
from rag.retrieval import reciprocal_rank_fusion, search
from rag.embeddings.base import Embeder
from rag.generators.base import Generator
from rag.reranker.base import Reranker
from rag.chunker import Chunker, ParentChildChunker


class HybridRAGPipeline(RAGPipeline):
    def __init__(
        self,
        sparse_embedder: Embeder | None = None,
        dense_embedder: Embeder | None = None,
        generator: Generator | None = None,
        chunker: Chunker | ParentChildChunker | None = None,
        reranker: Reranker | None = None,
        *,
        top_k: int = 5,
        chunk_size: int = 512,
        overlap: int = 50,
        rrf_k: int = 60,
        use_reranker: bool = True,
        use_hyde: bool = True,
        verbose: bool = True,
    ):
        from rag.embeddings import BM25Embeder, SentenceTransformerEmbeder

        super().__init__(generator=generator, chunker=chunker, top_k=top_k,
                         chunk_size=chunk_size, overlap=overlap)

        self.sparse_embedder = sparse_embedder or BM25Embeder()
        self.dense_embedder = dense_embedder or SentenceTransformerEmbeder()
        self.embedder = self.sparse_embedder  # base-class compat
        self.reranker = reranker or Reranker()
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.use_hyde = use_hyde
        self.verbose = verbose
        self.sparse_embeddings: list = []
        self.dense_embeddings: list = []

    def index(self, documents, source_names=None):
        all_chunks, sources, metadatas = self._prepare_chunks_and_sources(documents, source_names)
        self.chunks = all_chunks
        self.sources = sources
        self.metadatas = metadatas
        self.sparse_embedder.build_vocabulary(all_chunks)
        self.dense_embedder.build_vocabulary(all_chunks)
        self.vocab = self.sparse_embedder.vocabulary
        self.idf = getattr(self.sparse_embedder, "idf", [])
        self.sparse_embeddings = [self.sparse_embedder.embed(c) for c in all_chunks]
        self.dense_embeddings = [self.dense_embedder.embed(c) for c in all_chunks]
        return len(self.chunks)

    def _retrieve(self, question, top_k):
        retrieval_query = question
        if self.use_hyde:
            from rag.generators import ClaudeGenerator
            hyde_generator = ClaudeGenerator()
            if self.verbose:
                print(f"\n  [HyDE] Received: '{question}'")
            retrieval_query = hyde_generator.hyde_with_llm(question)
            if self.verbose:
                print(f"  [HyDE] Hypothetical doc: \"{retrieval_query}\"")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_sparse = ex.submit(self.sparse_embedder.embed, retrieval_query)
            f_dense = ex.submit(self.dense_embedder.embed, retrieval_query)
            sparse_query_emb, dense_query_emb = f_sparse.result(), f_dense.result()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            sparse_raw = ex.submit(search, sparse_query_emb, self.sparse_embeddings, 50).result()
            dense_raw = ex.submit(search, dense_query_emb, self.dense_embeddings, 50).result()

        fused = reciprocal_rank_fusion([dense_raw, sparse_raw], k=self.rrf_k)

        if self.use_reranker:
            fused = self.reranker.rerank(question, fused, self.chunks)

        return [
            {
                "chunk": self.chunks[idx],
                "score": score,
                "source": self.sources[idx],
                "chunk_position": (self.metadatas[idx] if idx < len(self.metadatas) else {}).get("chunk_position", "unknown"),
            }
            for idx, score in fused[:top_k]
        ]
