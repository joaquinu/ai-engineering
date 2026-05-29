

def build_rag_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    return f"""Answer the question based ONLY on the following context.
    If the context doesn't contain enough information, say "I don't have enough information to answer that."

    Context:
    {context}

    Question: {query}

    Answer:"""