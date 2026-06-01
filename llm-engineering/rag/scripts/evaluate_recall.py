from rag.sample_documents import SAMPLE_DOCUMENTS
from rag.pipeline import RAGPipeline, ChromaRAGPipeline
from rag.evaluation import evaluate_retrieval_recall

EVAL_SET = [
    ("What is the refund window for enterprise customers?", [0]),
    ("How much does the Professional plan cost?", [1]),
    ("What encryption does Acme use?", [2]),
    ("What are the API rate limits?", [3]),
    ("What happens if uptime falls below the SLA?", [4]),
]

if __name__ == "__main__":
    pipeline = RAGPipeline(embedder_type="tfidf")
    pipeline.index(SAMPLE_DOCUMENTS)
    avg_recall, results = evaluate_retrieval_recall(EVAL_SET, pipeline._retrieve)
    print(f"Average Recall@5: {avg_recall:.3f}")
    for r in results:
        print(f"  [{r['recall']:.2f}] {r['query'][:60]}")
