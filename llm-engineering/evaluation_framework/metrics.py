import random
import math
import hashlib
import re

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    # Splits by alphanumeric sequences and keeps single punctuation marks
    return re.findall(r'\w+|[^\w\s]', text.lower())

def rouge_l_score(reference, hypothesis):
    """
    Computes the ROUGE-L F1-score between reference and hypothesis texts.
    
    ROUGE-L measures the Longest Common Subsequence (LCS) between the two
    tokenized texts using dynamic programming. It calculates precision and recall
    based on the LCS length, and returns the harmonic mean (F1-score).
    
    Args:
        reference (str): The gold-standard target text.
        hypothesis (str): The candidate text to evaluate.
        
    Returns:
        float: The ROUGE-L F1-score rounded to 4 decimal places.
    """
    if not reference or not hypothesis:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    m = len(ref_tokens)
    n = len(hyp_tokens)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    if lcs_length == 0:
        return 0.0

    precision = lcs_length / n
    recall = lcs_length / m
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def word_overlap_score(reference, hypothesis):
    """
    Computes the Word Overlap Score between reference and hypothesis texts.
    
    Args:
        reference (str): The gold-standard target text.
        hypothesis (str): The candidate text to evaluate.
        
    Returns:
        float: The Word Overlap Score rounded to 4 decimal places.
    """
    if not reference or not hypothesis:
        return 0.0
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    intersection = ref_words & hyp_words
    union = ref_words | hyp_words
    return round(len(intersection) / len(union), 4) if union else 0.0


_word_vectors = {}

COMMON_WORDS = [
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
]

def _generate_vector(word: str) -> list:
    h = hashlib.sha256(word.encode('utf-8')).digest()
    state = random.Random(h)
    vec = [state.gauss(0.0, 1.0) for _ in range(50)]
    sq_sum = sum(x**2 for x in vec)
    norm = math.sqrt(sq_sum)
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

for word in COMMON_WORDS:
    _word_vectors[word] = _generate_vector(word)

def get_word_vector(word: str) -> list:
    word = word.lower().strip()
    if word not in _word_vectors:
        _word_vectors[word] = _generate_vector(word)
    return _word_vectors[word]

def cosine_similarity(v1: list, v2: list) -> float:
    return sum(x * y for x, y in zip(v1, v2))

def bert_score(reference: str, hypothesis: str) -> dict:
    """
    Computes a simplified BERTScore (Precision, Recall, and F1) between 
    reference and hypothesis texts using word embedding cosine similarity.
    
    Args:
        reference (str): The gold-standard target text.
        hypothesis (str): The candidate text to evaluate.
        
    Returns:
        dict: A dictionary containing precision, recall, and f1 scores.
    """
    if not reference or not hypothesis:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    
    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    ref_vectors = [get_word_vector(t) for t in ref_tokens]
    hyp_vectors = [get_word_vector(t) for t in hyp_tokens]
    
    sim_matrix = []
    for r_vec in ref_vectors:
        row = []
        for h_vec in hyp_vectors:
            row.append(cosine_similarity(r_vec, h_vec))
        sim_matrix.append(row)
        
    recall_sum = 0.0
    for i in range(len(ref_tokens)):
        max_sim = max(sim_matrix[i])
        recall_sum += max_sim
    recall = recall_sum / len(ref_tokens)
    
    precision_sum = 0.0
    for j in range(len(hyp_tokens)):
        max_sim = max(sim_matrix[i][j] for i in range(len(ref_tokens)))
        precision_sum += max_sim
    precision = precision_sum / len(hyp_tokens)
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

