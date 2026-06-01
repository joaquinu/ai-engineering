from rag.pipeline import RAGPipeline
from demo_runner import run_demo_pipeline

if __name__ == "__main__":
    run_demo_pipeline(lambda **kw: RAGPipeline(embedder_type="bm25", **kw), pipeline_name="BM25", is_chroma=False)
