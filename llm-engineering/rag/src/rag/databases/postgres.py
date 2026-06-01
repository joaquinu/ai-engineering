import os
import re
import json
import psycopg2
from rag.databases.base import Database


class PostgresDB(Database):
    def __init__(self, table_name="default", host=None, database=None, user=None, password=None, port=None):
        super().__init__()
        if not re.match(r'^[a-zA-Z0-9_]+$', table_name):
            raise ValueError(f"Invalid table name: '{table_name}'. Only alphanumeric characters and underscores are allowed.")
        self.table_name = table_name
        self.host = host or os.environ.get("POSTGRES_HOST") or os.environ.get("PGHOST") or "localhost"
        self.port = port or os.environ.get("POSTGRES_PORT") or os.environ.get("PGPORT") or "5432"
        self.database = database or os.environ.get("POSTGRES_DB") or os.environ.get("PGDATABASE") or "postgres"
        self.user = user or os.environ.get("POSTGRES_USER") or os.environ.get("PGUSER") or "postgres"
        self.password = password or os.environ.get("POSTGRES_PASSWORD") or os.environ.get("PGPASSWORD") or ""
        self.conn = None

    def _get_connection(self):
        if self.conn is None or self.conn.closed != 0:
            self.conn = psycopg2.connect(
                host=self.host, port=self.port, database=self.database,
                user=self.user, password=self.password, connect_timeout=3,
            )
            self.conn.autocommit = True
        return self.conn

    def _ensure_table(self, dimension):
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id VARCHAR(255) PRIMARY KEY,
                    document TEXT NOT NULL,
                    embedding VECTOR({dimension}),
                    metadata JSONB
                );
            """)

    def add(self, chunks, ids, embeddings, metadatas=None):
        if not chunks:
            return
        dimension = len(embeddings[0])
        self._ensure_table(dimension)
        conn = self._get_connection()
        with conn.cursor() as cur:
            for i, (chunk, c_id, emb) in enumerate(zip(chunks, ids, embeddings)):
                meta = metadatas[i] if metadatas and i < len(metadatas) else None
                meta_json = json.dumps(meta) if meta else None
                emb_str = f"[{','.join(map(str, emb))}]"
                cur.execute(f"""
                    INSERT INTO {self.table_name} (id, document, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        document = EXCLUDED.document,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                """, (str(c_id), chunk, emb_str, meta_json))

    def query(self, query_embedding, n_results=100):
        dimension = len(query_embedding)
        self._ensure_table(dimension)
        conn = self._get_connection()
        query_embedding_str = f"[{','.join(map(str, query_embedding))}]"
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, document, metadata, 1 - (embedding <=> %s) AS similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s
                LIMIT %s;
            """, (query_embedding_str, query_embedding_str, n_results))
            return [{"id": r[0], "document": r[1], "metadata": r[2] or {}, "score": r[3] or 0.0} for r in cur.fetchall()]

    def delete(self, ids):
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s);", ([str(i) for i in ids],))

    def get_all(self):
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, document, metadata FROM {self.table_name};")
            return [{"id": r[0], "document": r[1], "metadata": r[2] or {}} for r in cur.fetchall()]

    def get(self, ids):
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, document, metadata FROM {self.table_name} WHERE id = ANY(%s);", ([str(i) for i in ids],))
            return [{"id": r[0], "document": r[1], "metadata": r[2] or {}} for r in cur.fetchall()]

    def update(self, ids, chunks, embeddings, metadatas=None):
        self.add(chunks, ids, embeddings, metadatas)
