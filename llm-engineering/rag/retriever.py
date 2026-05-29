import math
from collections import Counter

class Chunker:
    def __init__(self, max_tokens=512, overlap=50):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[list[str]]:
        tokens = text.split()
        chunks = []
        for i in range(0, len(tokens), self.max_tokens - self.overlap):
            chunks.append(tokens[i:i + self.max_tokens])
        return chunks

class Embeder:
    def __init__(self, chunker, embedder, vector_store):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

class TFIDFEmbeder:
    def __init__(self, chunker):
        self.chunker = chunker
        self.vocabulary = None
        self.idf = None

    def build_vocabulary(self, documents):
        vocab = set()
        for doc in documents:
            if isinstance(doc, list):
                words = [w.lower() for w in doc]
            else:
                words = doc.lower().split()
            vocab.update(words)
        self.vocabulary = sorted(vocab)

    def compute_tf(self, text: str, vocab=None) -> list[float]:
        v = vocab if vocab is not None else self.vocabulary
        if isinstance(text, list):
            words = [w.lower() for w in text]
        else:
            words = text.lower().split()
        count = Counter(words)
        total = len(words) if words else 1
        return [count.get(word, 0) / total for word in v]

    def compute_idf(self, documents):
        n = len(documents)
        idf = []
        for word in self.vocabulary:
            doc_count = 0
            for doc in documents:
                if isinstance(doc, list):
                    words = [w.lower() for w in doc]
                else:
                    words = doc.lower().split()
                if word in words:
                    doc_count += 1
            idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
        self.idf = idf
    
    def tfidf_embed(self, text: str, vocab=None, idf=None) -> list[float]:
        v = vocab if vocab is not None else self.vocabulary
        i = idf if idf is not None else self.idf
        tf = self.compute_tf(text, v)
        return [t * val for t, val in zip(tf, i)]
    
    def cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query_embedding, stored_embeddings, top_k=5):
        scores = []
        for i, emb in enumerate(stored_embeddings):
            sim = self.cosine_similarity(query_embedding, emb)
            scores.append((i, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def embed(self, text: str):
        tf = self.compute_tf(text)
        return [t * i for t, i in zip(tf, self.idf)]