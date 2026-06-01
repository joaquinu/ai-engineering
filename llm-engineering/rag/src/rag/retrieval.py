import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(query_embedding: list[float], stored_embeddings: list[list[float]], top_k: int = 5) -> list[tuple[int, float]]:
    scores = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(stored_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
