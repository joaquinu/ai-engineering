import chromadb
from rag.databases.base import Database


class ChromaDB(Database):
    def __init__(self, collection="default"):
        super().__init__()
        self.client = chromadb.Client()
        try:
            self.collection = self.client.get_collection(name=collection)
        except Exception:
            self.collection = self.client.create_collection(name=collection)

    def add(self, chunks, ids, metadatas=None):
        self.collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    def query(self, query_text, n_results=100):
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def search(self, query_text: str, n_results: int = 5) -> list[dict]:
        raw = self.query(query_text, n_results)
        if not raw or not raw.get("documents"):
            return []
        docs = raw["documents"][0]
        ids = raw["ids"][0]
        metas = (raw.get("metadatas") or [[None] * len(docs)])[0]
        dists = (raw.get("distances") or [[0.0] * len(docs)])[0]
        return [
            {"id": c_id, "document": doc, "score": 1 - dist if dist <= 1.0 else 0.0, "metadata": meta or {}}
            for doc, c_id, meta, dist in zip(docs, ids, metas, dists)
        ]

    def delete(self, ids):
        self.collection.delete(ids=ids)

    def get_all(self):
        return self.collection.get()

    def get(self, ids):
        return self.collection.get(ids=ids)

    def update(self, ids, chunks, metadatas=None):
        self.collection.update(ids=ids, documents=chunks, metadatas=metadatas)
