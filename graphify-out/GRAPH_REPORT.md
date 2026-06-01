# Graph Report - .  (2026-06-01)

## Corpus Check
- Corpus is ~8,206 words - fits in a single context window. You may not need a graph.

## Summary
- 428 nodes · 626 edges · 34 communities (20 shown, 14 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 91 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_BM25 Core Algorithm|BM25 Core Algorithm]]
- [[_COMMUNITY_Database Abstraction|Database Abstraction]]
- [[_COMMUNITY_Embedder Implementations|Embedder Implementations]]
- [[_COMMUNITY_Chroma Pipeline & API|Chroma Pipeline & API]]
- [[_COMMUNITY_PostgresDB Backend|PostgresDB Backend]]
- [[_COMMUNITY_Demo Runner & Utilities|Demo Runner & Utilities]]
- [[_COMMUNITY_RAG Pipeline Base|RAG Pipeline Base]]
- [[_COMMUNITY_Conversational Pipeline|Conversational Pipeline]]
- [[_COMMUNITY_Retrieval Primitives Tests|Retrieval Primitives Tests]]
- [[_COMMUNITY_Prompt Builder Tests|Prompt Builder Tests]]
- [[_COMMUNITY_Embedder Semantic Layer|Embedder Semantic Layer]]
- [[_COMMUNITY_MCP Server|MCP Server]]
- [[_COMMUNITY_TF-IDF Embedder|TF-IDF Embedder]]
- [[_COMMUNITY_Hybrid Pipeline & HyDE|Hybrid Pipeline & HyDE]]
- [[_COMMUNITY_Simple Generator|Simple Generator]]
- [[_COMMUNITY_Pipeline Unit Tests|Pipeline Unit Tests]]
- [[_COMMUNITY_Generator Base & OpenAI|Generator Base & OpenAI]]
- [[_COMMUNITY_Chunker & Embedder Tests|Chunker & Embedder Tests]]
- [[_COMMUNITY_Claude Generator & HyDE|Claude Generator & HyDE]]
- [[_COMMUNITY_Conversational Tests|Conversational Tests]]
- [[_COMMUNITY_Database Lazy Init|Database Lazy Init]]
- [[_COMMUNITY_Reranker Layer|Reranker Layer]]
- [[_COMMUNITY_Prompt Builder Nodes|Prompt Builder Nodes]]
- [[_COMMUNITY_Evaluation Metrics|Evaluation Metrics]]
- [[_COMMUNITY_Sample Data|Sample Data]]
- [[_COMMUNITY_Embeddings Init|Embeddings Init]]
- [[_COMMUNITY_Generators Init|Generators Init]]
- [[_COMMUNITY_Pipeline Init|Pipeline Init]]
- [[_COMMUNITY_Qdrant Pipeline|Qdrant Pipeline]]
- [[_COMMUNITY_HyDE Tests|HyDE Tests]]
- [[_COMMUNITY_Pipeline Tests|Pipeline Tests]]
- [[_COMMUNITY_Postgres Tests|Postgres Tests]]

## God Nodes (most connected - your core abstractions)
1. `Chunker` - 22 edges
2. `PostgresDB` - 21 edges
3. `BM25Embeder` - 14 edges
4. `RAGPipeline Base Class` - 14 edges
5. `SimpleGenerator` - 12 edges
6. `TestTFIDFEmbeder` - 12 edges
7. `Database` - 11 edges
8. `ChromaDB` - 11 edges
9. `QdrantDB` - 11 edges
10. `BM25` - 11 edges

## Surprising Connections (you probably didn't know these)
- `TestConversationalRAGPipeline` --uses--> `SimpleGenerator`  [INFERRED]
  tests/test_pipelines.py → src/rag/generators/simple.py
- `TestHybridRAGPipeline` --uses--> `SimpleGenerator`  [INFERRED]
  tests/test_pipelines.py → src/rag/generators/simple.py
- `TestRAGPipeline` --uses--> `SimpleGenerator`  [INFERRED]
  tests/test_pipelines.py → src/rag/generators/simple.py
- `run_conversational_test()` --calls--> `ConversationalRAGPipeline`  [INFERRED]
  tests/test_conversational.py → src/rag/pipeline/conversational.py
- `TestBinaryBOWEmbeder` --uses--> `Chunker`  [INFERRED]
  tests/test_embedders.py → src/rag/tools.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Vector Database Backend Implementations** — chromadb_impl, qdrant_impl, postgres_impl [INFERRED 0.95]
- **Scripts Using Demo Runner** — chroma_run, qdrant_run, sentence_transformer_run, simple_bm25_run, simple_hybrid_run, simple_main [EXTRACTED 1.00]
- **RAG External Service Interfaces (API + MCP)** — rag_api, mcp_server, rag_pipeline_base [INFERRED 0.85]
- **Sparse/Lexical Embedders** — embeddings_tfidf_TFIDFEmbeder, embeddings_bow_BinaryBOWEmbeder, embeddings_bm25_BM25Embeder [INFERRED 0.95]
- **RAG Pipeline Implementations** — pipeline_chromadb_ChromaRAGPipeline, pipeline_hybrid_HybridRAGPipeline, pipeline_postgres_PostgresRAGPipeline, pipeline_conversational_ConversationalRAGPipeline [EXTRACTED 1.00]
- **LLM Generator Backends** — generators_claude_ClaudeGenerator, generators_openai_OpenAIGenerator, generators_simple_SimpleGenerator [EXTRACTED 1.00]
- **Prompt Builder Functions** — tools_build_rag_prompt, tools_build_attributed_rag_prompt, tools_build_conversational_prompt [INFERRED 0.95]
- **Evaluation Metric Functions** — tools_evaluate_faithfulness, tools_evaluate_retrieval_recall, test_evaluation [INFERRED 0.85]
- **Reranker Implementation Hierarchy** — reranker_base_Reranker, reranker_cohere_CohereReranker, reranker_init [EXTRACTED 1.00]

## Communities (34 total, 14 thin omitted)

### Community 0 - "BM25 Core Algorithm"
Cohesion: 0.06
Nodes (8): BM25, BM25Embeder, float, str, TestBinaryBOWEmbeder, TestBM25, TestBM25Embeder, TestTFIDFEmbeder

### Community 1 - "Database Abstraction"
Cohesion: 0.06
Nodes (6): Database, ChromaDB, QdrantDB, ChromaRAGPipeline, QdrantRAGPipeline, RAGPipeline

### Community 2 - "Embedder Implementations"
Cohesion: 0.06
Nodes (14): Embeder, BinaryBOWEmbeder, SentenceTransformerEmbeder, int, Reranker, CohereReranker, float, str (+6 more)

### Community 3 - "Chroma Pipeline & API"
Cohesion: 0.10
Nodes (29): BaseModel, ChromaRAGPipeline, Chroma Run Script, Compare Embeddings Script, Demo Runner, Evaluate Faithfulness Run Script, Evaluate Recall Script, HybridRAGPipeline (+21 more)

### Community 4 - "PostgresDB Backend"
Cohesion: 0.10
Nodes (5): PostgresDB, PostgresRAGPipeline, TestPostgresDBLive, TestPostgresDBMock, TestPostgresRAGPipeline

### Community 5 - "Demo Runner & Utilities"
Cohesion: 0.09
Nodes (7): run_demo_pipeline(), cosine_similarity(), evaluate_faithfulness(), evaluate_retrieval_recall(), search(), TestEvaluationMetrics, TestCosineSimilarity

### Community 6 - "RAG Pipeline Base"
Cohesion: 0.11
Nodes (6): RAGPipeline, Chunker, ParentChildChunker, str, TestChunker, TestParentChildChunking

### Community 7 - "Conversational Pipeline"
Cohesion: 0.18
Nodes (5): ConversationalRAGPipeline, build_conversational_prompt(), Integration smoke test for ConversationalRAGPipeline., run_conversational_test(), TestBuildConversationalPrompt

### Community 8 - "Retrieval Primitives Tests"
Cohesion: 0.17
Nodes (3): reciprocal_rank_fusion(), TestReciprocalRankFusion, TestSearch

### Community 9 - "Prompt Builder Tests"
Cohesion: 0.23
Nodes (4): build_attributed_rag_prompt(), build_rag_prompt(), TestBuildAttributedRagPrompt, TestBuildRagPrompt

### Community 10 - "Embedder Semantic Layer"
Cohesion: 0.23
Nodes (16): Embeder Base Class, BM25 Algorithm, BM25 Embeder, Binary BOW Embeder, SentenceTransformer Embeder, TF-IDF Embeder, Generator Base Class, Claude Generator (+8 more)

### Community 11 - "MCP Server"
Cohesion: 0.21
Nodes (11): Any, call_tool(), _ensure_pipeline(), list_tools(), main(), MCP (Model Context Protocol) server exposing RAG as tools for Claude and other a, _run(), RAGPipeline (+3 more)

### Community 12 - "TF-IDF Embedder"
Cohesion: 0.27
Nodes (3): TFIDFEmbeder, float, str

### Community 17 - "Chunker & Embedder Tests"
Cohesion: 0.25
Nodes (9): Test Chunker, Test Embedders, Test Parent-Child Chunking, Test Retrieval Primitives, Chunker, ParentChildChunker, cosine_similarity, reciprocal_rank_fusion (+1 more)

### Community 20 - "Database Lazy Init"
Cohesion: 0.73
Nodes (4): ChromaDB Implementation, Database Abstract Base Class, PostgresDB Implementation, QdrantDB Implementation

### Community 21 - "Reranker Layer"
Cohesion: 0.60
Nodes (3): Reranker (Base), CohereReranker, Test Rerankers

### Community 22 - "Prompt Builder Nodes"
Cohesion: 0.83
Nodes (4): Test Prompt Builders, build_attributed_rag_prompt, build_conversational_prompt, build_rag_prompt

### Community 23 - "Evaluation Metrics"
Cohesion: 1.00
Nodes (3): Test Evaluation Metrics, evaluate_faithfulness, evaluate_retrieval_recall

## Knowledge Gaps
- **27 isolated node(s):** `str`, `float`, `str`, `float`, `str` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `HybridRAGPipeline` connect `Hybrid Pipeline & HyDE` to `MCP Server`, `Database Abstraction`, `Embedder Implementations`, `Chroma Pipeline & API`?**
  _High betweenness centrality (0.225) - this node is a cross-community bridge._
- **Why does `_get_pipeline()` connect `Chroma Pipeline & API` to `Database Abstraction`, `PostgresDB Backend`, `Hybrid Pipeline & HyDE`, `Conversational Pipeline`?**
  _High betweenness centrality (0.141) - this node is a cross-community bridge._
- **Why does `Chunker` connect `RAG Pipeline Base` to `BM25 Core Algorithm`, `Embedder Implementations`, `TF-IDF Embedder`, `Demo Runner & Utilities`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Chunker` (e.g. with `RAGPipeline` and `TestChunker`) actually correct?**
  _`Chunker` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `PostgresDB` (e.g. with `Database` and `PostgresRAGPipeline`) actually correct?**
  _`PostgresDB` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `BM25Embeder` (e.g. with `.__init__()` and `.__init__()`) actually correct?**
  _`BM25Embeder` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `SimpleGenerator` (e.g. with `ClaudeGenerator` and `.generate()`) actually correct?**
  _`SimpleGenerator` has 8 INFERRED edges - model-reasoned connections that need verification._