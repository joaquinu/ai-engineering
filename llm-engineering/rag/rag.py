from retriever import Chunker
from retriever import TFIDFEmbeder
from augment import build_rag_prompt
import anthropic

class RAGPipeline:
    def __init__(self, chunk_size=512, overlap=50, top_k=5, generator_type = "simple"):
        self.top_k = top_k
        self.chunks = []
        self.sources = []
        self.embeddings = []
        self.vocab = []
        self.idf = []
        self.chunker = Chunker(chunk_size, overlap)
        self.embedder = TFIDFEmbeder(self.chunker)
        self.generator_type = generator_type

    def index(self, documents, source_names=None):
        all_chunks = []
        sources = []
        
        if source_names is None or len(source_names) != len(documents):
            default_sources = [
                "refund-policy.md",
                "product-overview.md",
                "security.md",
                "api-docs.md",
                "uptime-sla.md"
            ]
            if len(documents) == len(default_sources):
                source_names = default_sources
            else:
                source_names = [f"doc_{i}.md" for i in range(len(documents))]
        
        for doc, source in zip(documents, source_names):
            doc_chunks = self.chunker.chunk_text(doc)
            for chunk in doc_chunks:
                chunk_str = " ".join(chunk)
                all_chunks.append(chunk_str)
                sources.append(source)
                
        self.chunks = all_chunks
        self.sources = sources
        self.embedder.build_vocabulary(all_chunks)
        self.embedder.compute_idf(all_chunks)
        self.vocab = self.embedder.vocabulary
        self.idf = self.embedder.idf
        self.embeddings = [
            self.embedder.embed(chunk)
            for chunk in self.chunks
        ]
        return len(self.chunks)

    def query(self, question, top_k=5):
        query_emb = self.embedder.tfidf_embed(question, self.vocab, self.idf)
        results = self.embedder.search(query_emb, self.embeddings, top_k)
        
        retrieved_list = []
        for idx, score in results:
            retrieved_list.append({
                "chunk": self.chunks[idx],
                "score": score,
                "source": self.sources[idx]
            })
            
        chunks_for_prompt = [r["chunk"] for r in retrieved_list]
        prompt = build_rag_prompt(question, chunks_for_prompt)
        
        if self.generator_type == "simple":
            generator = SimpleGenerator()
        elif self.generator_type == "claude":
            generator = ClaudeGenerator()
        else:
            generator = SimpleGenerator()
        answer = generator.generate(prompt, chunks_for_prompt)
        
        return {
            "answer": answer,
            "retrieved": retrieved_list,
            "prompt": prompt
        }

class Generator:
    def __init__(self, generator=None):
        self.generator = generator

    def generate(self, prompt):
        if self.generator is not None:
            return self.generator(prompt)
        return ""

class SimpleGenerator(Generator):
    def __init__(self, generator=None):
        self.generator = generator

    def generate(self, prompt, retrieved_chunks):
        query_words = set(prompt.lower().split("question:")[-1].split())
        best_sentence = ""
        best_score = 0
        for chunk in retrieved_chunks:
            for sentence in chunk.split("."):
                sentence = sentence.strip()
                if not sentence:
                    continue
                words = set(sentence.lower().split())
                overlap = len(query_words & words)
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence
        return best_sentence if best_sentence else "I don't have enough information."

class ClaudeGenerator(Generator, model="claude-3-5-haiku-20241022"):
    def __init__(self, model=model):
        self.client = anthropic.Anthropic()
        self.model = model
    
    def generate(self, prompt, retrieved_chunks):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
        