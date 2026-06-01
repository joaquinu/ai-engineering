from rag.embeddings import BM25Embeder, SentenceTransformerEmbeder
from rag.pipeline import HybridRAGPipeline
from demo_runner import run_demo_pipeline

if __name__ == "__main__":
    run_demo_pipeline(
        lambda **kw: HybridRAGPipeline(sparse_embedder=BM25Embeder(), dense_embedder=SentenceTransformerEmbeder(), **kw),
        pipeline_name="Hybrid",
        is_chroma=False,
    )
