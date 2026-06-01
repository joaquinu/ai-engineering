from rag.pipeline import ChromaRAGPipeline
from demo_runner import run_demo_pipeline

if __name__ == "__main__":
    run_demo_pipeline(ChromaRAGPipeline, pipeline_name="ChromaDB", is_chroma=True)
