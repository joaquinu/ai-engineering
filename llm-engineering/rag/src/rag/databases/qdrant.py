from rag.databases.base import Database


class QdrantDB(Database):
    def __init__(self, collection="default"):
        super().__init__()
        from qdrant_client import QdrantClient
        self.client = QdrantClient(":memory:")
        self.collection_name = collection

    def add(self, chunks, ids, metadatas=None):
        self.client.add(collection_name=self.collection_name, documents=chunks, metadata=metadatas, ids=ids)

    def query(self, query_text, n_results=100):
        return self.client.query(collection_name=self.collection_name, query_text=query_text, limit=n_results)

    def search(self, query_text: str, n_results: int = 5) -> list[dict]:
        return [
            {"id": r.id, "document": r.document, "score": r.score, "metadata": r.metadata or {}}
            for r in self.query(query_text, n_results)
        ]

    def delete(self, ids):
        self.client.delete(collection_name=self.collection_name, points_selector=ids)

    def get_all(self):
        res, _ = self.client.scroll(collection_name=self.collection_name, limit=10000)
        return res

    def get(self, ids):
        return self.client.retrieve(collection_name=self.collection_name, ids=ids)

    def update(self, ids, chunks, metadatas=None):
        self.add(chunks, ids, metadatas)
