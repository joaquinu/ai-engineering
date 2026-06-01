from rag.pipeline.chromadb import ChromaRAGPipeline
from rag.prompts import build_conversational_prompt
from rag.generators import SimpleGenerator


class ConversationalRAGPipeline(ChromaRAGPipeline):
    def __init__(self, chunk_size=512, overlap=50, top_k=5, generator_type="simple",
                 collection_name="conversational_rag"):
        super().__init__(chunk_size=chunk_size, overlap=overlap, top_k=top_k,
                         generator_type=generator_type, collection_name=collection_name)
        self.history = []

    def clear_history(self):
        self.history = []

    def reformulate_query(self, question):
        if not self.history:
            return question

        if hasattr(self, "generator") and not isinstance(self.generator, SimpleGenerator):
            history_str = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.history])
            prompt = f"""Given the following conversation history and a follow-up question, reformulate the follow-up question into a standalone, search-friendly query that captures the user's intent. Do not answer the question, just return the reformulated question.

Conversation History:
{history_str}

Follow-up Question: {question}

Standalone Query:"""
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

    def query(self, question, top_k=None):
        retrieval_query = self.reformulate_query(question)
        retrieved_list = self._retrieve(retrieval_query, top_k or self.top_k)
        prompt = build_conversational_prompt(question, retrieved_list, self.history)
        answer = self.generator.generate(prompt, [r["chunk"] for r in retrieved_list])
        self.history.append({"user": question, "assistant": answer})
        if len(self.history) > 3:
            self.history.pop(0)
        return {"answer": answer, "retrieved": retrieved_list, "prompt": prompt}
