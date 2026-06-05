def build_rag_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
    )
    return f"""Answer the question based ONLY on the following context.
    If the context doesn't contain enough information, say "I don't have enough information to answer that."

    Context:
    {context}

    Question: {query}

    Answer:"""


def build_attributed_rag_prompt(query: str, retrieved_list: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source {i+1}] Document: {r.get('source', 'unknown')} ({r.get('chunk_position', 'unknown')})\n{r.get('chunk', '')}"
        for i, r in enumerate(retrieved_list)
    )
    return f"""Answer the question based ONLY on the following context.
    Cite the sources you used by referencing their tag (e.g. [Source 1], [Source 2]) at the end of the sentence or paragraph.
    If the context doesn't contain enough information, say "I don't have enough information to answer that."

    Context:
    {context}

    Question: {query}

    Answer:"""


def build_conversational_prompt(query: str, retrieved_list: list[dict], history: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source {i+1}] Document: {r.get('source', 'unknown')} ({r.get('chunk_position', 'unknown')})\n{r.get('chunk', '')}"
        for i, r in enumerate(retrieved_list)
    )
    history_str = ""
    if history:
        history_str = "\n\nConversation History:\n"
        for exchange in history:
            history_str += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"

    return f"""Answer the question based ONLY on the following context and conversation history.
    Cite the sources you used by referencing their tag (e.g. [Source 1], [Source 2]) at the end of the sentence or paragraph.
    If the context doesn't contain enough information, say "I don't have enough information to answer that."

    Context:
    {context}
    {history_str}
    Question: {query}

    Answer:"""
