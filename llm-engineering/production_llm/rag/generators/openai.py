from openai import OpenAI
from rag.generators.base import Generator

client = OpenAI()


class OpenAIGenerator(Generator):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__()
        self.model = model

    def embed(self, text):
        response = client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def generate(self, prompt, retrieved_chunks=None):
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content
