from rag.sample_documents import SAMPLE_DOCUMENTS
from rag.pipeline.factory import build_pipeline

if __name__ == "__main__":
    for embedder in ["tfidf", "bow", "bm25"]:
        p = build_pipeline(embedder=embedder)
        p.index(SAMPLE_DOCUMENTS)
        result = p.query("What is the refund policy?")
        print(f"[{embedder.upper()}] {result['answer'][:80]}")
