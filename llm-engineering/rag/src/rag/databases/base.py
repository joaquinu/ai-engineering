class Database:
    def __init__(self, database=None):
        self.client = None

    def add(self, chunks, ids):
        pass

    def query(self, query_text, n_results):
        pass

    def search(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Return results as a normalized list of {id, document, score, metadata} dicts.

        Managed-embedding backends (ChromaDB, Qdrant) implement this.
        Embedding-owned backends (PostgresDB) use query(embedding, n_results) instead.
        """
        return []

    def delete(self, ids):
        pass

    def get_all(self):
        pass

    def get(self, ids):
        pass

    def update(self, ids, chunks):
        pass
