import os
import unittest
from unittest.mock import MagicMock, patch, ANY
from rag.databases.postgres import PostgresDB
from rag.pipeline.postgres import PostgresRAGPipeline

SAMPLE_DOCS = [
    "The Starter plan costs $29 per month.",
    "Refund requests must be submitted within 30 days of purchase.",
    "Security features include AES-256 encryption.",
]


class TestPostgresDBMock(unittest.TestCase):

    def test_table_name_validation(self):
        with self.assertRaises(ValueError):
            PostgresDB(table_name="semi;colon")
        with self.assertRaises(ValueError):
            PostgresDB(table_name="spaces not allowed")

    @patch("rag.databases.postgres.psycopg2.connect")
    def test_mock_add(self, mock_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.closed = 0

        db = PostgresDB(table_name="test_table")
        db.add(["chunk 1", "chunk 2"], ["id1", "id2"], [[0.1, 0.2], [0.3, 0.4]], metadatas=[{"source": "doc1"}, {"source": "doc2"}])

        mock_cur.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS vector;")
        mock_cur.execute.assert_any_call(ANY, ("id1", "chunk 1", "[0.1,0.2]", '{"source": "doc1"}'))

    @patch("rag.databases.postgres.psycopg2.connect")
    def test_mock_query(self, mock_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.closed = 0
        mock_cur.fetchall.return_value = [("id1", "chunk 1", {"source": "doc1"}, 0.95)]

        db = PostgresDB(table_name="test_table")
        results = db.query(query_embedding=[0.1, 0.2], n_results=1)
        mock_cur.execute.assert_any_call(ANY, ("[0.1,0.2]", "[0.1,0.2]", 1))
        self.assertEqual(results[0]["id"], "id1")
        self.assertEqual(results[0]["score"], 0.95)

    @patch("rag.databases.postgres.psycopg2.connect")
    def test_mock_delete_and_get(self, mock_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.closed = 0

        db = PostgresDB(table_name="test_table")
        db.delete(["id1", "id2"])
        mock_cur.execute.assert_any_call(ANY, (["id1", "id2"],))

        mock_cur.fetchall.return_value = [("id1", "chunk 1", {"source": "doc1"})]
        db.get(["id1"])
        mock_cur.execute.assert_any_call(ANY, (["id1"],))


class TestPostgresDBLive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_name = "postgres_live_test_" + str(os.getpid())
        cls.db = None
        cls.skip_tests = False
        try:
            cls.db = PostgresDB(table_name=cls.db_name)
            cls.db._get_connection()
        except Exception as e:
            print(f"\n[INFO] Skipping Live PostgreSQL tests: {e}")
            cls.skip_tests = True

    def setUp(self):
        if self.skip_tests:
            self.skipTest("Live PostgreSQL database is not accessible")

    def tearDown(self):
        if self.db:
            try:
                with self.db._get_connection().cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {self.db_name};")
            except Exception:
                pass

    def test_live_lifecycle(self):
        chunks = ["Enterprise plan includes custom SLA", "Standard plan is cheaper"]
        embeddings = [[0.1] * 384, [0.9] * 384]
        self.db.add(chunks, ["c1", "c2"], embeddings, metadatas=[{"source": "live"}, {"source": "live"}])
        self.assertEqual(len(self.db.get_all()), 2)
        results = self.db.query([0.12] * 384, n_results=1)
        self.assertEqual(len(results), 1)
        self.db.delete(["c1"])
        self.assertEqual(len(self.db.get_all()), 1)


class TestPostgresRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("rag.pipeline.postgres.PostgresDB")
        self.mock_db_class = self.patcher.start()
        self.mock_db = MagicMock()
        self.mock_db_class.return_value = self.mock_db
        self.pipeline = PostgresRAGPipeline(chunk_size=30, overlap=5, top_k=2, generator_type="simple", embedder_type="bow", collection_name="mock_collection", verbose=False)

    def tearDown(self):
        self.patcher.stop()

    def test_pipeline_index(self):
        self.assertGreater(self.pipeline.index(SAMPLE_DOCS), 0)
        self.mock_db.add.assert_called_once()

    def test_pipeline_retrieve(self):
        self.mock_db.query.return_value = [{"id": "chunk_0", "document": "The Starter plan costs $29 per month.", "metadata": {"source": "live"}, "score": 0.99}]
        results = self.pipeline._retrieve("starter plan", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Starter", results[0]["chunk"])

    def test_pipeline_end_to_end(self):
        self.mock_db.query.return_value = [{"id": "chunk_0", "document": "The Starter plan costs $29 per month.", "metadata": {"source": "live"}, "score": 0.99}]
        self.pipeline.index(SAMPLE_DOCS)
        result = self.pipeline.query("How much does the Starter plan cost?")
        self.assertIn("$29", result["answer"])


if __name__ == "__main__":
    unittest.main()
