"""
MCP (Model Context Protocol) server exposing RAG as tools for Claude and other agents.

Run:
    python -m rag.mcp_server
    # or via entry point:
    rag-mcp

Add to claude_desktop_config.json:
    {"mcpServers": {"rag": {"command": "rag-mcp"}}}
"""
from __future__ import annotations

from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from rag.pipeline.base import RAGPipeline

_pipeline: RAGPipeline | None = None


def _ensure_pipeline(pipeline_type: str = "simple", embedder_type: str = "tfidf",
                     generator_type: str = "simple") -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        from rag.pipeline.factory import build_pipeline, build_hybrid_pipeline, _make_generator
        if pipeline_type == "chroma":
            from rag.databases.chromadb import ChromaDB
            from rag.pipeline.vectordb import VectorDBPipeline
            _pipeline = VectorDBPipeline(db=ChromaDB("mcp_collection"),
                                         generator=_make_generator(generator_type))
        elif pipeline_type == "hybrid":
            _pipeline = build_hybrid_pipeline(generator=generator_type)
        else:
            _pipeline = build_pipeline(embedder=embedder_type, generator=generator_type)
    return _pipeline


server = Server("rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="index_documents",
            description="Index a list of text documents into the RAG pipeline so they can be queried.",
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {"type": "array", "items": {"type": "string"}},
                    "source_names": {"type": "array", "items": {"type": "string"}},
                    "pipeline_type": {"type": "string", "enum": ["simple", "chroma", "hybrid"], "default": "simple"},
                    "embedder_type": {"type": "string", "enum": ["tfidf", "bow", "bm25", "sentence_transformers"], "default": "tfidf"},
                    "generator_type": {"type": "string", "enum": ["simple", "claude"], "default": "simple"},
                },
                "required": ["documents"],
            },
        ),
        Tool(
            name="query",
            description="Ask a question against the indexed documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="clear_index",
            description="Reset the pipeline and discard all indexed documents.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    global _pipeline

    if name == "index_documents":
        pipeline = _ensure_pipeline(
            pipeline_type=arguments.get("pipeline_type", "simple"),
            embedder_type=arguments.get("embedder_type", "tfidf"),
            generator_type=arguments.get("generator_type", "simple"),
        )
        n = pipeline.index(arguments["documents"], arguments.get("source_names"))
        return [TextContent(type="text", text=f"Indexed {n} chunks from {len(arguments['documents'])} document(s).")]

    if name == "query":
        if _pipeline is None:
            return [TextContent(type="text", text="No documents indexed yet. Call index_documents first.")]
        result = _pipeline.query(arguments["question"], top_k=arguments.get("top_k", 5))
        sources = [f"[{r['source']} {r['chunk_position']}] (score={r['score']:.3f})" for r in result["retrieved"]]
        return [TextContent(type="text", text=f"Answer: {result['answer']}\n\nSources:\n" + "\n".join(sources))]

    if name == "clear_index":
        _pipeline = None
        return [TextContent(type="text", text="Index cleared.")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    import asyncio
    asyncio.run(_run())


if __name__ == "__main__":
    main()
