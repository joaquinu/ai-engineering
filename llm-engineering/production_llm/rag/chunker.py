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


class ParentChildChunker:
    def __init__(self, parent_size=512, parent_overlap=50, child_size=128, child_overlap=10):
        self.parent_chunker = Chunker(parent_size, parent_overlap)
        self.child_chunker = Chunker(child_size, child_overlap)

    def chunk_document(self, text: str) -> list[dict]:
        results = []
        for p_idx, p_chunk in enumerate(self.parent_chunker.chunk_text(text)):
            p_text = " ".join(p_chunk)
            for c_idx, c_chunk in enumerate(self.child_chunker.chunk_text(p_text)):
                results.append({
                    "child_chunk": c_chunk,
                    "parent_text": p_text,
                    "parent_index": p_idx,
                    "child_index": c_idx,
                })
        return results
