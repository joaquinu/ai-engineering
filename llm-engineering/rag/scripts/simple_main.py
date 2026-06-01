from rag.pipeline import RAGPipeline
from demo_runner import run_demo_pipeline

if __name__ == "__main__":
    run_demo_pipeline(RAGPipeline, pipeline_name="TF-IDF", is_chroma=False)
