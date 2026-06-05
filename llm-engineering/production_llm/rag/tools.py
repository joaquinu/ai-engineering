# Backward-compatible re-exports — prefer the direct sub-modules in new code:
#   rag.chunker    — Chunker, ParentChildChunker
#   rag.prompts    — build_*_prompt
#   rag.retrieval  — cosine_similarity, search, reciprocal_rank_fusion
#   rag.evaluation — evaluate_faithfulness, evaluate_retrieval_recall
from rag.chunker import Chunker, ParentChildChunker
from rag.prompts import build_rag_prompt, build_attributed_rag_prompt, build_conversational_prompt
from rag.retrieval import cosine_similarity, search, reciprocal_rank_fusion
from rag.evaluation import evaluate_faithfulness, evaluate_retrieval_recall

__all__ = [
    "Chunker", "ParentChildChunker",
    "build_rag_prompt", "build_attributed_rag_prompt", "build_conversational_prompt",
    "cosine_similarity", "search", "reciprocal_rank_fusion",
    "evaluate_faithfulness", "evaluate_retrieval_recall",
]
