import concurrent.futures
from rag.pipeline.base import RAGPipeline
from rag.retrieval import reciprocal_rank_fusion, search
from rag.embeddings import BM25Embeder


class HybridRAGPipeline(RAGPipeline):
    def __init__(self, chunk_size=512, overlap=50, top_k=5, generator_type="simple", rrf_k=60,
                 sparse_embedder_type="bm25", dense_embedder_type="sentence_transformers",
                 use_reranker=True, use_hyde=True, reranker_type="simple", verbose=True):
        super().__init__(chunk_size=chunk_size, overlap=overlap, top_k=top_k, generator_type=generator_type)
        self.use_hyde = use_hyde
        self.verbose = verbose
        self.sparse_embedder_type = sparse_embedder_type
        self.dense_embedder_type = dense_embedder_type
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker

        if sparse_embedder_type == "bow":
            from rag.embeddings import BinaryBOWEmbeder
            self.sparse_embedder = BinaryBOWEmbeder(self.chunker)
        elif sparse_embedder_type == "tfidf":
            from rag.embeddings import TFIDFEmbeder
            self.sparse_embedder = TFIDFEmbeder(self.chunker)
        else:
            self.sparse_embedder = BM25Embeder(self.chunker)

        if dense_embedder_type == "bow":
            from rag.embeddings import BinaryBOWEmbeder
            self.dense_embedder = BinaryBOWEmbeder(self.chunker)
        elif dense_embedder_type == "tfidf":
            from rag.embeddings import TFIDFEmbeder
            self.dense_embedder = TFIDFEmbeder(self.chunker)
        else:
            from rag.embeddings import SentenceTransformerEmbeder
            self.dense_embedder = SentenceTransformerEmbeder(self.chunker)

        self.embedder = self.sparse_embedder

        if reranker_type == "cohere":
            from rag.reranker import CohereReranker
            self.reranker = CohereReranker()
        else:
            from rag.reranker import Reranker
            self.reranker = Reranker()

        self.sparse_embeddings = []
        self.dense_embeddings = []

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
            sparse_query_emb = f_sparse.result()
            dense_query_emb = f_dense.result()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_sparse_s = ex.submit(search, sparse_query_emb, self.sparse_embeddings, 50)
            f_dense_s = ex.submit(search, dense_query_emb, self.dense_embeddings, 50)
            sparse_raw = f_sparse_s.result()
            dense_raw = f_dense_s.result()

        fused = reciprocal_rank_fusion([dense_raw, sparse_raw], k=self.rrf_k)

        if self.use_reranker:
            fused = self.reranker.rerank(question, fused, self.chunks)

        retrieved_list = []
        for idx, score in fused[:top_k]:
            meta = self.metadatas[idx] if idx < len(self.metadatas) else {}
            retrieved_list.append({
                "chunk": self.chunks[idx],
                "score": score,
                "source": self.sources[idx],
                "chunk_position": meta.get("chunk_position", "unknown"),
            })
        return retrieved_list
