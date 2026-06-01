from rag.sample_documents import SAMPLE_DOCUMENTS
from rag.pipeline.factory import build_pipeline
from rag.evaluation import evaluate_faithfulness

if __name__ == "__main__":
    pipeline = build_pipeline(embedder="tfidf")
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
