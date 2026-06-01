from rag.embeddings import BM25Embeder
from rag.pipeline import RAGPipeline
from demo_runner import run_demo_pipeline

if __name__ == "__main__":
    run_demo_pipeline(lambda **kw: RAGPipeline(embedder=BM25Embeder(), **kw), pipeline_name="BM25", is_chroma=False)
