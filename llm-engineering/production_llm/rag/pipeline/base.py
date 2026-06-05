from rag.chunker import Chunker, ParentChildChunker
from rag.prompts import build_attributed_rag_prompt
from rag.retrieval import search
from rag.generators import ClaudeGenerator, SimpleGenerator
from rag.embeddings import TFIDFEmbeder, BinaryBOWEmbeder
from rag.embeddings.base import Embeder
from rag.generators.base import Generator


class RAGPipeline:
    def __init__(
        self,
        embedder: Embeder | None = None,
        generator: Generator | None = None,
        chunker: Chunker | ParentChildChunker | None = None,
        *,
        top_k: int = 5,
        chunk_size: int = 512,
        overlap: int = 50,
    ):
        self.top_k = top_k
        self.chunker = chunker or Chunker(chunk_size, overlap)
        self.embedder = embedder or TFIDFEmbeder()
        self.generator = generator or SimpleGenerator()

        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.metadatas: list[dict] = []
        self.embeddings: list[list[float]] = []
        self.vocab: list[str] = []
        self.idf: list[float] = []

    def _prepare_chunks_and_sources(self, documents, source_names=None):
        all_chunks, sources, metadatas = [], [], []

        if source_names is None or len(source_names) != len(documents):
            default_sources = ["refund-policy.md", "product-overview.md", "security.md", "api-docs.md", "uptime-sla.md"]
            source_names = default_sources if len(documents) == len(default_sources) else [f"doc_{i}.md" for i in range(len(documents))]

        for doc, source in zip(documents, source_names):
            if isinstance(self.chunker, ParentChildChunker):
                for item in self.chunker.chunk_document(doc):
                    all_chunks.append(" ".join(item["child_chunk"]))
                    sources.append(source)
                    metadatas.append({
                        "source": source,
                        "chunk_position": f"parent chunk {item['parent_index'] + 1}, child chunk {item['child_index'] + 1}",
                        "chunk_index": len(all_chunks) - 1,
                        "parent_chunk": item["parent_text"],
                    })
            else:
                doc_chunks = self.chunker.chunk_text(doc)
                for idx, chunk in enumerate(doc_chunks):
                    all_chunks.append(" ".join(chunk))
                    sources.append(source)
                    metadatas.append({"source": source, "chunk_position": f"chunk {idx + 1} of {len(doc_chunks)}", "chunk_index": idx})

        return all_chunks, sources, metadatas

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
        embeddings = [self.embedder.embed(chunk) for chunk in self.chunks]
        self._store_index(embeddings)
        return len(self.chunks)

    def _store_index(self, embeddings: list) -> None:
        """Persist chunk embeddings. Default: keep in memory. Override for external stores."""
        self.embeddings = embeddings

    def _retrieve(self, question, top_k):
        if hasattr(self.embedder, "tfidf_embed"):
            query_emb = self.embedder.tfidf_embed(question, self.vocab, self.idf)
        else:
            query_emb = self.embedder.embed(question)

        retrieved_list = []
        for idx, score in search(query_emb, self.embeddings, top_k):
            meta = self.metadatas[idx] if idx < len(self.metadatas) else {}
            retrieved_list.append({
                "chunk": meta.get("parent_chunk", self.chunks[idx]),
                "score": score,
                "source": self.sources[idx],
                "chunk_position": meta.get("chunk_position", "unknown"),
            })
        return retrieved_list

    def query(self, question, top_k=None):
        retrieved_list = self._retrieve(question, top_k or self.top_k)
        prompt = build_attributed_rag_prompt(question, retrieved_list)
        answer = self.generator.generate(prompt, [r["chunk"] for r in retrieved_list])
        return {"answer": answer, "retrieved": retrieved_list, "prompt": prompt}
