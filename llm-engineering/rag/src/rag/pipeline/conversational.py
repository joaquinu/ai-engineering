from rag.pipeline.base import RAGPipeline
from rag.prompts import build_conversational_prompt
from rag.generators import SimpleGenerator
from rag.generators.base import Generator
from rag.chunker import Chunker, ParentChildChunker


class ConversationalRAGPipeline:
    """Adds conversation history to any RAGPipeline backend.

    Usage:
        # Default: ChromaDB backend
        rag = ConversationalRAGPipeline(collection_name="chat")

        # Any backend:
        rag = ConversationalRAGPipeline(pipeline=VectorDBPipeline(db=QdrantDB("chat")))
        rag = ConversationalRAGPipeline(pipeline=RAGPipeline(embedder=BM25Embeder()))
    """

    def __init__(
        self,
        pipeline: RAGPipeline | None = None,
        collection_name: str = "conversational_rag",
        generator: Generator | None = None,
        chunker: Chunker | ParentChildChunker | None = None,
        *,
        top_k: int = 5,
        chunk_size: int = 512,
        overlap: int = 50,
    ):
        if pipeline is None:
            from rag.databases.chromadb import ChromaDB
            from rag.pipeline.vectordb import VectorDBPipeline
            pipeline = VectorDBPipeline(
                db=ChromaDB(collection_name),
                generator=generator,
                chunker=chunker,
                top_k=top_k,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        self.pipeline = pipeline
        self.history: list[dict] = []

    # ── delegation ────────────────────────────────────────────────────────────

    def index(self, documents, source_names=None):
        return self.pipeline.index(documents, source_names)

    def _retrieve(self, question, top_k):
        return self.pipeline._retrieve(question, top_k)

    @property
    def top_k(self):
        return self.pipeline.top_k

    @property
    def generator(self):
        return self.pipeline.generator

    # ── conversational logic ───────────────────────────────────────────────────

    def clear_history(self):
        self.history = []

    def reformulate_query(self, question: str) -> str:
        if not self.history:
            return question

        if not isinstance(self.generator, SimpleGenerator):
            history_str = "\n".join(
                f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.history
            )
            prompt = (
                f"Given the following conversation history and a follow-up question, "
                f"reformulate the follow-up into a standalone, search-friendly query. "
                f"Return only the reformulated question.\n\n"
                f"Conversation History:\n{history_str}\n\n"
                f"Follow-up Question: {question}\n\nStandalone Query:"
            )
            try:
                reformulated = self.generator.generate(prompt).strip().strip('"').strip("'")
                if reformulated:
                    return reformulated
            except Exception:
                pass

        root_query = self.history[0]["user"]
        follow_up_indicators = ["what about", "and ", "how about", "why", "where", "who", "when",
                                 "cost", "price", "is it", "does it", "can i", "are they", "which"]
        q_lower = question.lower()
        if len(question.split()) < 6 or any(q_lower.startswith(ind) for ind in follow_up_indicators):
            return f"{root_query} {question}"
        return question

    def query(self, question: str, top_k: int | None = None) -> dict:
        retrieval_query = self.reformulate_query(question)
        retrieved_list = self._retrieve(retrieval_query, top_k or self.top_k)
        prompt = build_conversational_prompt(question, retrieved_list, self.history)
        answer = self.generator.generate(prompt, [r["chunk"] for r in retrieved_list])
        self.history.append({"user": question, "assistant": answer})
        if len(self.history) > 3:
            self.history.pop(0)
        return {"answer": answer, "retrieved": retrieved_list, "prompt": prompt}
