import anthropic
from rag.generators.base import Generator


class ClaudeGenerator(Generator):
    def __init__(self, model="claude-3-5-haiku-20241022"):
        super().__init__()
        self.model = model
        try:
            self.client = anthropic.Anthropic()
        except Exception:
            self.client = None

    def generate(self, prompt, retrieved_chunks=None):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"\n  [WARN] LLM Generation API call failed ({e}). Falling back to local offline heuristic extractor.")
            from rag.generators.simple import SimpleGenerator
            return SimpleGenerator().generate(prompt, retrieved_chunks or [])

    def hyde_with_llm(self, query):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}",
                }],
            )
            return response.content[0].text
        except Exception as e:
            print(f"\n  [WARN] HyDE API call failed ({e}). Falling back to local offline simulation.")
            words = [w for w in query.lower().replace("?", "").split() if len(w) > 3]
            keywords = ", ".join(words[:4])
            return f"Acme Corporation provides comprehensive customer support, pricing plans, and policy terms. Regarding {query.lower()}, the system specifies detailed guidelines, encryption standards, and customer options for {keywords}."
