from typing import Callable


def evaluate_faithfulness(answer: str, retrieved_chunks: list[str]) -> tuple[float, list[str]]:
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                  "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
    context = " ".join(retrieved_chunks).lower()
    grounded = 0
    ungrounded = []

    for sentence in answer_sentences:
        content_words = set(sentence.lower().split()) - stop_words
        if not content_words:
            grounded += 1
            continue
        if sum(1 for w in content_words if w in context) / len(content_words) >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    return grounded / len(answer_sentences), ungrounded


def evaluate_retrieval_recall(
    queries_with_relevant: list[tuple[str, list[int]]],
    retrieval_fn: Callable,
    k: int = 5,
) -> tuple[float, list[dict]]:
    total_recall = 0.0
    results = []
    for query, relevant_indices in queries_with_relevant:
        retrieved_indices = {idx for idx, _ in retrieval_fn(query, k)}
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({"query": query, "recall": recall, "hits": hits, "total_relevant": len(relevant_set)})
    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0.0
    return avg_recall, results
