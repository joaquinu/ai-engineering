from rag.sample_documents import SAMPLE_DOCUMENTS
from rag.pipeline import build_pipeline
from rag.evaluation import evaluate_retrieval_recall

EVAL_SET = [
    ("What is the refund window for enterprise customers?", [0]),
    ("How much does the Professional plan cost?", [1]),
    ("What encryption does Acme use?", [2]),
    ("What are the API rate limits?", [3]),
    ("What happens if uptime falls below the SLA?", [4]),
]

if __name__ == "__main__":
    pipeline = build_pipeline(embedder="tfidf")
    pipeline.index(SAMPLE_DOCUMENTS)

    def get_indices(question, k):
        """Adapter to convert _retrieve results (dicts) to (index, score) tuples."""
        results = pipeline._retrieve(question, top_k=k)
        # Extract chunk_index from metadata if available, otherwise use source doc index
        indices = []
        for result in results:
            if "chunk_index" in result:
                idx = result["chunk_index"]
            else:
                idx = SAMPLE_DOCUMENTS.index(result["chunk"]) if result["chunk"] in SAMPLE_DOCUMENTS else 0
            indices.append((idx, result["score"]))
        return indices

    avg_recall, results = evaluate_retrieval_recall(EVAL_SET, get_indices)
    print(f"Average Recall@5: {avg_recall:.3f}")
    for r in results:
        print(f"  [{r['recall']:.2f}] {r['query'][:60]}")
