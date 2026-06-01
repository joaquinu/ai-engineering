from rag.embeddings import SentenceTransformerEmbeder
from rag.pipeline import RAGPipeline
from demo_runner import run_demo_pipeline

if __name__ == "__main__":
    run_demo_pipeline(lambda **kw: RAGPipeline(embedder=SentenceTransformerEmbeder(), **kw), pipeline_name="SentenceTransformers", is_chroma=False)
