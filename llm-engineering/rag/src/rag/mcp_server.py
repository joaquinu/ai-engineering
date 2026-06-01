"""
MCP (Model Context Protocol) server exposing RAG as tools for Claude and other agents.

Run:
    python -m rag.mcp_server
    # or via entry point:
    rag-mcp

Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "rag": {
          "command": "rag-mcp"
        }
      }
    }
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from rag.pipeline import RAGPipeline

# Single shared in-memory pipeline (swap for persistent store in production)
_pipeline: RAGPipeline | None = None


def _ensure_pipeline(pipeline_type: str = "simple", **kwargs) -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        if pipeline_type == "chroma":
            from rag.pipeline import ChromaRAGPipeline
            _pipeline = ChromaRAGPipeline(**kwargs)
        elif pipeline_type == "hybrid":
            from rag.pipeline import HybridRAGPipeline
            _pipeline = HybridRAGPipeline(**kwargs)
        else:
            _pipeline = RAGPipeline(**kwargs)
    return _pipeline


server = Server("rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="index_documents",
            description=(
                "Index a list of text documents into the RAG pipeline so they can be queried. "
                "Returns the number of chunks stored."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document texts to index.",
                    },
                    "source_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional display names for each document (same length as documents).",
                    },
                    "pipeline_type": {
                        "type": "string",
                        "enum": ["simple", "chroma", "hybrid"],
                        "default": "simple",
                        "description": "Which pipeline backend to use.",
                    },
                    "embedder_type": {
                        "type": "string",
                        "enum": ["tfidf", "bow", "bm25", "sentence_transformers"],
                        "default": "tfidf",
                    },
                    "generator_type": {
                        "type": "string",
                        "enum": ["simple", "claude"],
                        "default": "simple",
                    },
                },
                "required": ["documents"],
            },
        ),
        Tool(
            name="query",
            description=(
                "Ask a question against the indexed documents. "
                "Returns the answer and the retrieved source chunks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to answer."},
                    "top_k": {"type": "integer", "default": 5, "description": "Number of chunks to retrieve."},
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
        documents = arguments["documents"]
        source_names = arguments.get("source_names")
        pipeline_type = arguments.get("pipeline_type", "simple")
        embedder_type = arguments.get("embedder_type", "tfidf")
        generator_type = arguments.get("generator_type", "simple")
        pipeline = _ensure_pipeline(
            pipeline_type=pipeline_type,
            embedder_type=embedder_type,
            generator_type=generator_type,
        )
        n = pipeline.index(documents, source_names)
        return [TextContent(type="text", text=f"Indexed {n} chunks from {len(documents)} document(s).")]

    if name == "query":
        if _pipeline is None:
            return [TextContent(type="text", text="No documents indexed yet. Call index_documents first.")]
        question = arguments["question"]
        top_k = arguments.get("top_k", 5)
        result = _pipeline.query(question, top_k=top_k)
        sources = [f"[{r['source']} {r['chunk_position']}] (score={r['score']:.3f})" for r in result["retrieved"]]
        output = f"Answer: {result['answer']}\n\nSources:\n" + "\n".join(sources)
        return [TextContent(type="text", text=output)]

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
