"""Integration smoke test for ConversationalRAGPipeline."""
from rag.sample_documents import SAMPLE_DOCUMENTS
from rag.pipeline import ConversationalRAGPipeline


def run_conversational_test():
    print("=" * 70)
    print("TESTING CONVERSATION-AWARE RAG PIPELINE")
    print("=" * 70)

    rag = ConversationalRAGPipeline(chunk_size=50, overlap=10, top_k=3, collection_name="chat_test_collection")
    rag.index(SAMPLE_DOCUMENTS)
    print("Indexed documents.\n")

    for idx, query in enumerate(["How much does the Starter plan cost?", "What about the Professional plan?", "And Enterprise?"]):
        print(f"EXCHANGE {idx + 1}: USER: {query}")
        result = rag.query(query, top_k=2)
        print(f"ASSISTANT: {result['answer']}\n")


if __name__ == "__main__":
    run_conversational_test()
