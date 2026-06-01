import unittest
from rag.chunker import ParentChildChunker
from rag.pipeline import RAGPipeline, ChromaRAGPipeline
from rag.pipeline.factory import build_pipeline


class TestParentChildChunking(unittest.TestCase):

    def test_parent_child_chunker_logic(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunker = ParentChildChunker(parent_size=40, parent_overlap=10, child_size=10, child_overlap=2)
        doc_chunks = chunker.chunk_document(text)
        self.assertGreater(len(doc_chunks), 0)
        first_item = doc_chunks[0]
        self.assertIn("child_chunk", first_item)
        self.assertIn("parent_text", first_item)
        self.assertEqual(len(first_item["child_chunk"]), 10)
        child_str = " ".join(first_item["child_chunk"])
        self.assertIn(child_str, first_item["parent_text"])

    def test_in_memory_pipeline_with_parent_child(self):
        documents = ["This is a long document containing a section about refund policies. The refund policy is standard. All plans receive refunds within 30 days."]
        pipeline = build_pipeline(chunker="parent_child", chunk_size=20, overlap=5, top_k=1)
        num_chunks = pipeline.index(documents)
        self.assertGreater(num_chunks, 1)
        results = pipeline._retrieve("refund policy", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]["chunk"].split()), 10)

    def test_chroma_pipeline_with_parent_child(self):
        documents = ["This is a long document containing a section about refund policies. The refund policy is standard. All plans receive refunds within 30 days."]
        pipeline = ChromaRAGPipeline(
            collection_name="parent_child_test",
            chunker=ParentChildChunker(parent_size=20, parent_overlap=5, child_size=32, child_overlap=5),
            top_k=1,
        )
        num_chunks = pipeline.index(documents)
        self.assertGreater(num_chunks, 1)
        results = pipeline._retrieve("refund policy", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]["chunk"].split()), 10)


if __name__ == "__main__":
    unittest.main()
