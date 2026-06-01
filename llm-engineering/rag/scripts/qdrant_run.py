from rag.pipeline import QdrantRAGPipeline
from demo_runner import run_demo_pipeline

if __name__ == "__main__":
    run_demo_pipeline(QdrantRAGPipeline, pipeline_name="Qdrant", is_chroma=False)
