import unittest
from rag.chunker import Chunker


class TestChunker(unittest.TestCase):

    def test_single_chunk_when_text_fits(self):
        chunker = Chunker(max_tokens=10, overlap=2)
        chunks = chunker.chunk_text("one two three")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], ["one", "two", "three"])

    def test_exact_chunk_size_no_overlap(self):
        chunker = Chunker(max_tokens=4, overlap=0)
        chunks = chunker.chunk_text("a b c d e f g h")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], ["a", "b", "c", "d"])
        self.assertEqual(chunks[1], ["e", "f", "g", "h"])

    def test_overlap_creates_shared_tokens(self):
        chunker = Chunker(max_tokens=5, overlap=2)
        text = "w0 w1 w2 w3 w4 w5 w6"
        chunks = chunker.chunk_text(text)
        self.assertEqual(chunks[0][-2:], chunks[1][:2])

    def test_multiple_chunks_count(self):
        chunker = Chunker(max_tokens=4, overlap=1)
        text = " ".join(str(i) for i in range(12))
        self.assertEqual(len(chunker.chunk_text(text)), 4)

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(len(Chunker(max_tokens=10, overlap=2).chunk_text("")), 0)

    def test_single_word(self):
        chunks = Chunker(max_tokens=5, overlap=1).chunk_text("hello")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], ["hello"])

    def test_text_shorter_than_chunk_size(self):
        self.assertEqual(len(Chunker(max_tokens=100, overlap=20).chunk_text("short text here")), 1)

    def test_last_chunk_may_be_smaller(self):
        chunks = Chunker(max_tokens=4, overlap=0).chunk_text("a b c d e f")
        self.assertEqual(chunks[-1], ["e", "f"])


if __name__ == "__main__":
    unittest.main()
