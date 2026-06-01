import unittest
from rag.prompts import build_rag_prompt, build_attributed_rag_prompt, build_conversational_prompt


class TestBuildRagPrompt(unittest.TestCase):

    def test_contains_query(self):
        self.assertIn("What is the refund policy?", build_rag_prompt("What is the refund policy?", ["Context chunk."]))

    def test_contains_retrieved_chunk(self):
        chunk = "Customers get 30 days to request a refund."
        self.assertIn(chunk, build_rag_prompt("refund?", [chunk]))

    def test_multiple_chunks_numbered(self):
        prompt = build_rag_prompt("query", ["chunk one", "chunk two", "chunk three"])
        self.assertIn("[Source 1]", prompt)
        self.assertIn("[Source 3]", prompt)

    def test_empty_chunks_still_generates_prompt(self):
        prompt = build_rag_prompt("any question?", [])
        self.assertIn("any question?", prompt)
        self.assertIsInstance(prompt, str)

    def test_prompt_instructs_context_only(self):
        self.assertIn("ONLY", build_rag_prompt("q?", ["ctx"]))


class TestBuildAttributedRagPrompt(unittest.TestCase):

    def _make_retrieved(self, n=2):
        return [{"source": f"doc{i}.md", "chunk_position": f"chunk {i+1} of 5", "chunk": f"Content {i}"} for i in range(n)]

    def test_contains_source_tags(self):
        prompt = build_attributed_rag_prompt("What happened?", self._make_retrieved(2))
        self.assertIn("[Source 1]", prompt)
        self.assertIn("[Source 2]", prompt)

    def test_source_filenames_appear(self):
        prompt = build_attributed_rag_prompt("q?", self._make_retrieved(2))
        self.assertIn("doc0.md", prompt)

    def test_chunk_content_appears(self):
        self.assertIn("Content 0", build_attributed_rag_prompt("q?", self._make_retrieved(2)))

    def test_citation_instruction_present(self):
        prompt = build_attributed_rag_prompt("q?", self._make_retrieved())
        self.assertIn("Cite", prompt)

    def test_missing_source_key_uses_unknown(self):
        self.assertIn("unknown", build_attributed_rag_prompt("q?", [{"chunk": "some text"}]))


class TestBuildConversationalPrompt(unittest.TestCase):

    def _retrieved(self):
        return [{"source": "doc.md", "chunk_position": "chunk 1 of 3", "chunk": "The price is $99."}]

    def test_no_history_omits_header(self):
        self.assertNotIn("Conversation History:", build_conversational_prompt("How much?", self._retrieved(), history=[]))

    def test_with_history_includes_header(self):
        history = [{"user": "What plans exist?", "assistant": "We have Starter and Pro."}]
        self.assertIn("Conversation History:", build_conversational_prompt("Price?", self._retrieved(), history=history))

    def test_history_turns_appear(self):
        history = [{"user": "Tell me about plans.", "assistant": "We offer Starter, Pro, Enterprise."}]
        prompt = build_conversational_prompt("Price?", self._retrieved(), history=history)
        self.assertIn("Tell me about plans.", prompt)
        self.assertIn("We offer Starter, Pro, Enterprise.", prompt)

    def test_current_question_appears(self):
        self.assertIn("Current question here?", build_conversational_prompt("Current question here?", self._retrieved(), history=[]))


if __name__ == "__main__":
    unittest.main()
