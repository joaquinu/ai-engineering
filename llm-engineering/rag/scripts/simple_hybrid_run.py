from rag.pipeline import HybridRAGPipeline
from demo_runner import run_demo_pipeline

if __name__ == "__main__":
    run_demo_pipeline(
        lambda **kw: HybridRAGPipeline(sparse_embedder_type="bm25", dense_embedder_type="sentence_transformers", **kw),
        pipeline_name="Hybrid",
        is_chroma=False,
    )
