from rag.sample_documents import SAMPLE_DOCUMENTS
from rag.retrieval import cosine_similarity


def run_demo_pipeline(pipeline_class, pipeline_name="TF-IDF", is_chroma=False):
    print("=" * 60)
    print("STEP 1: Document Chunking")
    print("=" * 60)

    is_vector_db = pipeline_name in ["ChromaDB", "Qdrant"]

    if is_vector_db:
        rag = pipeline_class(chunk_size=30, overlap=10, collection_name=f"demo_step1_{pipeline_name.lower()}")
    else:
        rag = pipeline_class(chunk_size=30, overlap=10)

    sample = SAMPLE_DOCUMENTS[0]
    chunks = rag.chunker.chunk_text(sample)
    print(f"  Document length: {len(sample.split())} words")
    print(f"  Chunk size: 30 words, overlap: 10 words")
    print(f"  Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        chunk_str = " ".join(chunk) if isinstance(chunk, list) else chunk
        print(f"\n  Chunk {i}: ({len(chunk)} words)")
        print(f"    {chunk_str[:100]}...")
    if len(chunks) > 3:
        print("  ...")

    print("\n" + "=" * 60)
    if is_vector_db:
        print(f"STEP 2: {pipeline_name} Setup & Indexing")
    elif "hybrid" in pipeline_name.lower():
        print("STEP 2: Hybrid Search Setup & Parallel Indexing")
    elif "bm25" in pipeline_name.lower():
        print("STEP 2: BM25 Vocabulary & Index Setup")
    else:
        print("STEP 2: TF-IDF Embedding")
    print("=" * 60)

    if is_vector_db:
        rag_main = pipeline_class(chunk_size=50, overlap=10, top_k=3, collection_name=f"demo_main_{pipeline_name.lower()}")
        num_chunks = rag_main.index(SAMPLE_DOCUMENTS)
        print(f"  Successfully initialized {pipeline_name} Client (in-memory)")
        print(f"  Indexed {len(SAMPLE_DOCUMENTS)} documents into {num_chunks} chunks")
    else:
        rag_main = pipeline_class(chunk_size=50, overlap=10, top_k=3)
        mini_docs = [
            "The cat sat on the mat",
            "The dog sat on the rug",
            "Machine learning is a branch of artificial intelligence",
        ]
        rag_main.embedder.build_vocabulary(mini_docs)
        rag_main.embedder.compute_idf(mini_docs)
        print(f"  Vocabulary size: {len(rag_main.embedder.vocabulary)}")
        if "hybrid" in pipeline_name.lower():
            sparse_name = getattr(rag_main, "sparse_embedder_type", "bm25").upper()
            dense_name = getattr(rag_main, "dense_embedder_type", "sentence_transformers").upper()
            print(f"  Vocabulary & IDF built for {sparse_name} sparse index.")
            print(f"  Neural embeddings prepared for {dense_name} dense index.")
        elif "bm25" in pipeline_name.lower():
            print("  Vocabulary & IDF built for BM25 term index.")
        else:
            for word, score in sorted(zip(rag_main.embedder.vocabulary, rag_main.embedder.idf), key=lambda x: x[1], reverse=True)[:8]:
                print(f"    {word:20s} IDF={score:.3f}")
        num_chunks = rag_main.index(SAMPLE_DOCUMENTS)

    print("\n" + "=" * 60)
    print("STEP 3: Retrieval")
    print("=" * 60)

    test_query = "What is the refund policy for enterprise customers?"
    if is_vector_db:
        print(f"  Query: {test_query}")
        for i, r in enumerate(rag_main._retrieve(test_query, top_k=3)):
            preview = r["chunk"][:80].replace("\n", " ")
            metric = f"distance={r['distance']:.4f}" if "distance" in r else f"score={r['score']:.4f}"
            print(f"    Match {i+1}: [{r['source']}] {metric} | {preview}...")
    else:
        mini_docs = ["The cat sat on the mat", "The dog sat on the rug", "Machine learning is a branch of artificial intelligence"]
        rag_temp = pipeline_class(chunk_size=30, overlap=10)
        rag_temp.embedder.build_vocabulary(mini_docs)
        rag_temp.embedder.compute_idf(mini_docs)
        emb1 = rag_temp.embedder.tfidf_embed(mini_docs[0])
        emb2 = rag_temp.embedder.tfidf_embed(mini_docs[1])
        emb3 = rag_temp.embedder.tfidf_embed(mini_docs[2])
        print(f"  'cat on mat' vs 'dog on rug':       {cosine_similarity(emb1, emb2):.4f}")
        print(f"  'cat on mat' vs 'machine learning': {cosine_similarity(emb1, emb3):.4f}")

    print("\n" + "=" * 60)
    print(f"STEP 4: Full RAG Pipeline with {pipeline_name}")
    print("=" * 60)

    queries = [
        "What is the refund policy for enterprise customers?",
        "What are the API rate limits?",
        "How is customer data encrypted?",
        "What happens if uptime falls below the SLA?",
        "How much does the Professional plan cost?",
    ]
    for query in queries:
        print(f"\n  Query: {query}")
        result = rag_main.query(query, top_k=3)
        print(f"  Answer: {result['answer']}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Pipeline: {pipeline_name}")
    print(f"  Documents: {len(SAMPLE_DOCUMENTS)}, Chunks: {num_chunks}")
