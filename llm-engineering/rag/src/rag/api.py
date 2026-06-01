"""
FastAPI service exposing RAG as an HTTP API.

Run:
    uvicorn rag.api:app --reload
    # or via entry point:
    rag-api
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RAG Service", version="0.1.0")

# One pipeline instance per (pipeline_type, collection_name) kept in memory.
# For production swap this with a proper registry / persistent vector store.
_pipelines: dict[str, object] = {}


def _get_pipeline(pipeline_type: str, collection_name: str, **kwargs):
    key = f"{pipeline_type}:{collection_name}"
    if key not in _pipelines:
        if pipeline_type == "chroma":
            from rag.pipeline import ChromaRAGPipeline
            _pipelines[key] = ChromaRAGPipeline(collection_name=collection_name, **kwargs)
        elif pipeline_type == "qdrant":
            from rag.pipeline import QdrantRAGPipeline
            _pipelines[key] = QdrantRAGPipeline(collection_name=collection_name, **kwargs)
        elif pipeline_type == "postgres":
            from rag.pipeline import PostgresRAGPipeline
            _pipelines[key] = PostgresRAGPipeline(collection_name=collection_name, **kwargs)
        elif pipeline_type == "hybrid":
            from rag.pipeline import HybridRAGPipeline
            _pipelines[key] = HybridRAGPipeline(**kwargs)
        elif pipeline_type == "conversational":
            from rag.pipeline import ConversationalRAGPipeline
            _pipelines[key] = ConversationalRAGPipeline(collection_name=collection_name, **kwargs)
        else:
            from rag.pipeline import RAGPipeline
            _pipelines[key] = RAGPipeline(**kwargs)
    return _pipelines[key]


# ── request / response models ────────────────────────────────────────────────

class IndexRequest(BaseModel):
    documents: list[str]
    source_names: list[str] | None = None
    pipeline_type: Literal["simple", "chroma", "qdrant", "postgres", "hybrid", "conversational"] = "simple"
    collection_name: str = "default"
    chunk_size: int = 512
    overlap: int = 50
    embedder_type: str = "tfidf"
    generator_type: str = "simple"


class IndexResponse(BaseModel):
    chunks_indexed: int
    collection_name: str
    pipeline_type: str


class QueryRequest(BaseModel):
    question: str
    pipeline_type: Literal["simple", "chroma", "qdrant", "postgres", "hybrid", "conversational"] = "simple"
    collection_name: str = "default"
    top_k: int = 5


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


@app.post("/index", response_model=IndexResponse)
def index(req: IndexRequest):
    try:
        pipeline = _get_pipeline(
            req.pipeline_type,
            req.collection_name,
            chunk_size=req.chunk_size,
            overlap=req.overlap,
            embedder_type=req.embedder_type,
            generator_type=req.generator_type,
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


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("rag.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
