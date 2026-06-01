from rag.databases.base import Database


def __getattr__(name):
    if name == "ChromaDB":
        from rag.databases.chromadb import ChromaDB
        return ChromaDB
    if name == "QdrantDB":
        from rag.databases.qdrant import QdrantDB
        return QdrantDB
    if name == "PostgresDB":
        from rag.databases.postgres import PostgresDB
        return PostgresDB
    raise AttributeError(f"module 'rag.databases' has no attribute {name!r}")


__all__ = ["Database", "ChromaDB", "QdrantDB", "PostgresDB"]
