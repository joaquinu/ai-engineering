import re
from rag.generators.base import Generator


class SimpleGenerator(Generator):
    def __init__(self, generator=None):
        self.generator = generator

    def generate(self, prompt, retrieved_chunks=None):
        if retrieved_chunks is None:
            retrieved_chunks = []
        query_words = set(prompt.lower().split("question:")[-1].split())
        best_sentence = ""
        best_score = 0
        for chunk in retrieved_chunks:
            sentences = re.split(
                r'(?<!\d)(?<!\bCorp)(?<!\bInc)(?<!\bLtd)(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bMrs)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)\.(?!\d)(?=\s|$)',
                chunk,
            )
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                overlap = len(query_words & set(sentence.lower().split()))
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence
        return best_sentence if best_sentence else "I don't have enough information."
