"""Configuration for RAG service, read from environment variables."""
import os


class RAGConfig:
    """RAG service configuration."""

    # Default collection name for all pipelines
    DEFAULT_COLLECTION_NAME = os.environ.get("RAG_COLLECTION_NAME", "default")

    # Default embedder type: tfidf, sentence_transformers, bow, bm25
    DEFAULT_EMBEDDER_TYPE = os.environ.get("RAG_EMBEDDER_TYPE", "tfidf")

    # Default generator type: simple, claude, openai
    DEFAULT_GENERATOR_TYPE = os.environ.get("RAG_GENERATOR_TYPE", "simple")

    # Default pipeline type: simple, postgres, chroma, qdrant, hybrid, conversational
    DEFAULT_PIPELINE_TYPE = os.environ.get("RAG_PIPELINE_TYPE", "postgres")

    # RAG parameters
    CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))
    TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

    # Global retrieval augmentation features (available for all pipeline types)
    USE_RERANKER = os.environ.get("RAG_USE_RERANKER", "false").lower() == "true"
    USE_HYDE = os.environ.get("RAG_USE_HYDE", "false").lower() == "true"
    VERBOSE = os.environ.get("RAG_VERBOSE", "false").lower() == "true"

    # Hybrid pipeline parameters
    HYBRID_SPARSE_EMBEDDER = os.environ.get("RAG_HYBRID_SPARSE_EMBEDDER", "bm25")  # bm25, tfidf
    HYBRID_DENSE_EMBEDDER = os.environ.get("RAG_HYBRID_DENSE_EMBEDDER", "sentence_transformers")  # sentence_transformers
    HYBRID_RRF_K = int(os.environ.get("RAG_HYBRID_RRF_K", "60"))

    # Postgres configuration (used by PostgresDB)
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
    POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

    @classmethod
    def to_dict(cls):
        """Return all config as a dictionary."""
        return {
            "DEFAULT_COLLECTION_NAME": cls.DEFAULT_COLLECTION_NAME,
            "DEFAULT_EMBEDDER_TYPE": cls.DEFAULT_EMBEDDER_TYPE,
            "DEFAULT_GENERATOR_TYPE": cls.DEFAULT_GENERATOR_TYPE,
            "DEFAULT_PIPELINE_TYPE": cls.DEFAULT_PIPELINE_TYPE,
            "CHUNK_SIZE": cls.CHUNK_SIZE,
            "CHUNK_OVERLAP": cls.CHUNK_OVERLAP,
            "TOP_K": cls.TOP_K,
        }
