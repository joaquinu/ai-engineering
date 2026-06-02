"""Index sample docs into the postgres pgvector pipeline and run a test query."""
import argparse
import json
import urllib.request

BASE_URL = "http://localhost:8000"

DOCUMENTS = [
    "Acme Corp Refund Policy. All standard plan customers are eligible for a full refund within 30 days of purchase. Enterprise plan customers receive an extended 60-day refund window.",
    "Acme Corp Product Overview. Acme Corp offers three product tiers: Starter at $29/mo, Professional at $99/mo, and Enterprise starting at $500/mo for up to 50 users.",
    "Acme Corp Security Practices. Acme Corp maintains SOC 2 Type II compliance. All data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
    "Acme Corp API Documentation. The Acme API uses REST with JSON. Authentication is via Bearer tokens issued through OAuth 2.0. Rate limits vary by plan.",
    "Acme Corp Uptime and Reliability. Acme Corp guarantees 99.9% uptime for Professional plans and 99.99% for Enterprise plans, calculated monthly.",
]

SOURCE_NAMES = [
    "refund-policy.md",
    "product-overview.md",
    "security.md",
    "api-docs.md",
    "uptime-sla.md",
]


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get(path):
    req = urllib.request.Request(
        BASE_URL + path,
        method="GET",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Index and query the RAG pipeline")
    parser.add_argument("question", nargs="?", default="What is the refund policy for enterprise customers?",
                        help="Question to ask the RAG pipeline")
    args = parser.parse_args()

    # Check service configuration first to dynamically display the active pipeline/embedders
    config_result = get("/config")
    pipeline_type = config_result.get("DEFAULT_PIPELINE_TYPE", "postgres")
    embedder_type = config_result.get("DEFAULT_EMBEDDER_TYPE", "tfidf")

    if pipeline_type == "hybrid":
        sparse = config_result.get("HYBRID_SPARSE_EMBEDDER", "bm25")
        dense = config_result.get("HYBRID_DENSE_EMBEDDER", "sentence_transformers")
        pipeline_info = f"HybridRAGPipeline ({sparse} + {dense})"
    elif pipeline_type == "postgres":
        pipeline_info = f"PostgresRAGPipeline (base pipeline + pgvector + {embedder_type})"
    else:
        pipeline_info = f"{pipeline_type.capitalize()}RAGPipeline ({embedder_type})"

    print(f"Indexing sample documents via {pipeline_info}...")
    print("  Using service defaults (env-configured)...")
    result = post("/index", {
        "documents": DOCUMENTS,
        "source_names": SOURCE_NAMES,
    })
    print(f"  Indexed {result['chunks_indexed']} chunks into collection '{result['collection_name']}'")
    print(f"  Pipeline: {result['pipeline_type']}")

    print(f"\nService Configuration:")
    for key, value in config_result.items():
        print(f"  {key}: {value}")

    print(f"\nQuery: {args.question!r}")
    result = post("/query", {
        "question": args.question,
    })
    print(f"\nAnswer:\n  {result['answer']}")
    print("\nRetrieved:")
    for r in result["retrieved"]:
        print(f"  [{r['score']:.3f}] {r['source']}  {r['chunk'][:90]}...")


if __name__ == "__main__":
    main()
