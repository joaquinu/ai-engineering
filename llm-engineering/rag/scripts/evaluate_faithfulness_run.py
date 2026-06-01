from rag.sample_documents import SAMPLE_DOCUMENTS
from rag.pipeline import RAGPipeline
from rag.evaluation import evaluate_faithfulness

if __name__ == "__main__":
    pipeline = RAGPipeline(embedder_type="tfidf")
    pipeline.index(SAMPLE_DOCUMENTS)

    queries = [
        "What is the refund policy for enterprise customers?",
        "How is customer data encrypted?",
        "What are the API rate limits?",
    ]
    for query in queries:
        result = pipeline.query(query)
        score, ungrounded = evaluate_faithfulness(result["answer"], [r["chunk"] for r in result["retrieved"]])
        print(f"Query: {query}")
        print(f"  Faithfulness: {score:.2f}  Ungrounded sentences: {len(ungrounded)}")
