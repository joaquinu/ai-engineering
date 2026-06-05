"""
FastAPI service exposing RAG as an HTTP API.

Run:
    uvicorn rag.api:app --reload
    # or via entry point:
    rag-api

Configuration via environment variables:
    RAG_PIPELINE_TYPE      - default pipeline (postgres, chroma, qdrant, simple, etc.)
    RAG_COLLECTION_NAME    - default collection name
    RAG_EMBEDDER_TYPE      - default embedder (tfidf, sentence_transformers, bow, bm25)
    RAG_GENERATOR_TYPE     - default generator (simple, claude, openai)
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.config import RAGConfig

app = FastAPI(title="RAG Service", version="0.1.0")

_pipelines: dict[str, object] = {}


def _get_pipeline(pipeline_type: str, collection_name: str,
                  embedder_type: str = "tfidf", generator_type: str = "simple",
                  chunk_size: int = 512, overlap: int = 50, top_k: int = 5,
                  use_reranker: bool = False, use_hyde: bool = False, verbose: bool = False,
                  hybrid_sparse_embedder: str = "bm25",
                  hybrid_dense_embedder: str = "sentence_transformers",
                  hybrid_rrf_k: int = 60,
                  chunker_type: str = "standard",
                  child_size: int | None = None,
                  child_overlap: int | None = None):
    key = f"{pipeline_type}:{collection_name}"
    if key not in _pipelines:
        from rag.pipeline.factory import build_pipeline, build_hybrid_pipeline, _make_generator, _make_chunker

        chunker = _make_chunker(
            chunker_type, chunk_size, overlap,
            child_size=child_size, child_overlap=child_overlap,
        )

        if pipeline_type in ("chroma", "conversational"):
            from rag.databases.chromadb import ChromaDB
            from rag.pipeline.vectordb import VectorDBPipeline
            vdb = VectorDBPipeline(
                db=ChromaDB(collection_name),
                generator=_make_generator(generator_type),
                top_k=top_k, chunker=chunker,
            )
            if pipeline_type == "conversational":
                from rag.pipeline.conversational import ConversationalRAGPipeline
                _pipelines[key] = ConversationalRAGPipeline(pipeline=vdb)
            else:
                _pipelines[key] = vdb

        elif pipeline_type == "qdrant":
            from rag.databases.qdrant import QdrantDB
            from rag.pipeline.vectordb import VectorDBPipeline
            _pipelines[key] = VectorDBPipeline(
                db=QdrantDB(collection_name),
                generator=_make_generator(generator_type),
                top_k=top_k, chunker=chunker,
            )

        elif pipeline_type == "postgres":
            from rag.pipeline.postgres import PostgresRAGPipeline
            from rag.pipeline.factory import _make_embedder
            _pipelines[key] = PostgresRAGPipeline(
                embedder=_make_embedder(embedder_type),
                generator=_make_generator(generator_type),
                collection_name=collection_name,
                top_k=top_k, chunker=chunker,
            )

        elif pipeline_type == "hybrid":
            from rag.pipeline.factory import _make_generator, _make_embedder
            from rag.pipeline.hybrid import HybridRAGPipeline
            _pipelines[key] = HybridRAGPipeline(
                sparse_embedder=_make_embedder(hybrid_sparse_embedder),
                dense_embedder=_make_embedder(hybrid_dense_embedder),
                generator=_make_generator(generator_type),
                chunker=chunker,
                top_k=top_k,
                rrf_k=hybrid_rrf_k,
                use_reranker=use_reranker,
                use_hyde=use_hyde,
                verbose=verbose,
            )

        else:
            _pipelines[key] = build_pipeline(
                embedder=embedder_type, generator=generator_type,
                chunker=chunker_type, chunk_size=chunk_size, overlap=overlap, top_k=top_k,
                child_size=child_size, child_overlap=child_overlap,
            )

    # Apply global retrieval augmentation wrappers (for non-hybrid pipelines)
    pipeline = _pipelines[key]
    if pipeline_type != "hybrid" and (use_hyde or use_reranker):
        from rag.reranker.base import Reranker
        # Wrap the pipeline's _retrieve method with HyDE + reranking
        original_retrieve = pipeline._retrieve

        def augmented_retrieve(question, top_k):
            retrieval_query = question
            if use_hyde:
                from rag.generators import ClaudeGenerator
                hyde_gen = ClaudeGenerator()
                if verbose:
                    print(f"\n  [HyDE] Received: '{question}'")
                retrieval_query = hyde_gen.hyde_with_llm(question)
                if verbose:
                    print(f"  [HyDE] Hypothetical doc: \"{retrieval_query}\"")

            # Retrieve with expanded query
            results = original_retrieve(retrieval_query, 50)

            # Rerank if enabled
            if use_reranker:
                reranker = Reranker()
                chunks = [r["chunk"] for r in results]
                candidates = [(i, r["score"]) for i, r in enumerate(results)]
                reranked_scores = reranker.rerank(question, candidates, chunks)
                results = [
                    {**results[idx], "score": score}
                    for idx, score in reranked_scores
                ]

            return results[:top_k]

        pipeline._retrieve = augmented_retrieve

    return _pipelines[key]


# ── request / response models ────────────────────────────────────────────────

class IndexRequest(BaseModel):
    documents: list[str]
    source_names: list[str] | None = None
    pipeline_type: Literal["simple", "chroma", "qdrant", "postgres", "hybrid", "conversational"] = RAGConfig.DEFAULT_PIPELINE_TYPE
    collection_name: str = RAGConfig.DEFAULT_COLLECTION_NAME
    chunk_size: int = RAGConfig.CHUNK_SIZE
    overlap: int = RAGConfig.CHUNK_OVERLAP
    embedder_type: str = RAGConfig.DEFAULT_EMBEDDER_TYPE
    generator_type: str = RAGConfig.DEFAULT_GENERATOR_TYPE
    # Global retrieval augmentation (for all pipeline types)
    use_reranker: bool = RAGConfig.USE_RERANKER
    use_hyde: bool = RAGConfig.USE_HYDE
    verbose: bool = RAGConfig.VERBOSE
    # Hybrid pipeline parameters (optional)
    hybrid_sparse_embedder: str = RAGConfig.HYBRID_SPARSE_EMBEDDER
    hybrid_dense_embedder: str = RAGConfig.HYBRID_DENSE_EMBEDDER
    hybrid_rrf_k: int = RAGConfig.HYBRID_RRF_K
    # Chunker configuration (optional)
    chunker_type: str = RAGConfig.CHUNKER_TYPE
    child_size: int = RAGConfig.CHILD_SIZE
    child_overlap: int = RAGConfig.CHILD_OVERLAP


class IndexResponse(BaseModel):
    chunks_indexed: int
    collection_name: str
    pipeline_type: str


class QueryRequest(BaseModel):
    question: str
    pipeline_type: Literal["simple", "chroma", "qdrant", "postgres", "hybrid", "conversational"] = RAGConfig.DEFAULT_PIPELINE_TYPE
    collection_name: str = RAGConfig.DEFAULT_COLLECTION_NAME
    top_k: int = RAGConfig.TOP_K


class RetrievedChunk(BaseModel):
    chunk: str
    score: float
    source: str
    chunk_position: str


class QueryResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedChunk]


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    """Show the current RAG configuration from environment variables."""
    return RAGConfig.to_dict()


@app.post("/index", response_model=IndexResponse)
def index(req: IndexRequest):
    try:
        pipeline = _get_pipeline(
            req.pipeline_type, req.collection_name,
            embedder_type=req.embedder_type, generator_type=req.generator_type,
            chunk_size=req.chunk_size, overlap=req.overlap,
            use_reranker=req.use_reranker, use_hyde=req.use_hyde, verbose=req.verbose,
            hybrid_sparse_embedder=req.hybrid_sparse_embedder,
            hybrid_dense_embedder=req.hybrid_dense_embedder,
            hybrid_rrf_k=req.hybrid_rrf_k,
            chunker_type=req.chunker_type,
            child_size=req.child_size,
            child_overlap=req.child_overlap,
        )
        n = pipeline.index(req.documents, req.source_names)
        return IndexResponse(chunks_indexed=n, collection_name=req.collection_name, pipeline_type=req.pipeline_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    key = f"{req.pipeline_type}:{req.collection_name}"
    if key not in _pipelines:
        raise HTTPException(status_code=404, detail=f"Collection '{req.collection_name}' not indexed yet. Call /index first.")
    try:
        result = _pipelines[key].query(req.question, top_k=req.top_k)
        return QueryResponse(
            answer=result["answer"],
            retrieved=[RetrievedChunk(**{k: r.get(k, "") for k in RetrievedChunk.model_fields}) for r in result["retrieved"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection/{collection_name}")
def delete_collection(collection_name: str, pipeline_type: str = "simple"):
    key = f"{pipeline_type}:{collection_name}"
    if key in _pipelines:
        del _pipelines[key]
        return {"deleted": collection_name}
    raise HTTPException(status_code=404, detail="Collection not found.")


def main():
    import uvicorn
    uvicorn.run("rag.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
