from sample_documents import SAMPLE_DOCUMENTS
from rag import RAGPipeline

import copy




if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Document Chunking")
    print("=" * 60)
    gen_type="claude"

    rag = RAGPipeline(chunk_size=30, overlap=10, generator_type=gen_type)

    sample = SAMPLE_DOCUMENTS[0]
    chunks = rag.chunker.chunk_text(sample)
    print(f"  Document length: {len(sample.split())} words")
    print(f"  Chunk size: 30 words, overlap: 10 words")
    print(f"  Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i}: ({len(chunk)} words)")
        print(f"    {chunk[:100]}...")

    print("\n" + "=" * 60)
    print("STEP 2: TF-IDF Embedding")
    print("=" * 60)

    mini_docs = [
        "The cat sat on the mat",
        "The dog sat on the rug",
        "Machine learning is a branch of artificial intelligence"
    ]
    rag.embedder.build_vocabulary(mini_docs)
    rag.embedder.compute_idf(mini_docs)

    print(f"  Vocabulary size: {len(rag.embedder.vocabulary)}")
    print(f"  Sample words and IDF scores:")
    for word, score in sorted(zip(rag.embedder.vocabulary, rag.embedder.idf), key=lambda x: x[1], reverse=True)[:8]:
        print(f"    {word:20s} IDF={score:.3f}")

    rag1 = copy.deepcopy(rag)
    rag2 = copy.deepcopy(rag)
    rag3 = copy.deepcopy(rag)

    emb1 = rag1.embedder.tfidf_embed(mini_docs[0])
    emb2 = rag2.embedder.tfidf_embed(mini_docs[1])
    emb3 = rag3.embedder.tfidf_embed(mini_docs[2])

    print(f"\n  Embedding dimensions: {len(emb1)}")
    print(f"  Non-zero entries in 'cat sat on mat': {sum(1 for v in emb1 if v > 0)}")
    print(f"  Non-zero entries in 'dog sat on rug': {sum(1 for v in emb2 if v > 0)}")
    print(f"  Non-zero entries in 'machine learning': {sum(1 for v in emb3 if v > 0)}")

    print("\n" + "=" * 60)
    print("STEP 3: Cosine Similarity")
    print("=" * 60)

    sim_12 = rag.embedder.cosine_similarity(emb1, emb2)
    sim_13 = rag.embedder.cosine_similarity(emb1, emb3)
    sim_23 = rag.embedder.cosine_similarity(emb2, emb3)

    print(f"  'cat on mat' vs 'dog on rug':     {sim_12:.4f}  (similar structure)")
    print(f"  'cat on mat' vs 'machine learning': {sim_13:.4f}  (unrelated)")
    print(f"  'dog on rug' vs 'machine learning': {sim_23:.4f}  (unrelated)")
    print(f"\n  As expected: similar sentences score higher.")

    print("\n" + "=" * 60)
    print("STEP 4: Full RAG Pipeline")
    print("=" * 60)

    rag = RAGPipeline(chunk_size=50, overlap=10, top_k=3, generator_type=gen_type)
    source_names = [
        "refund-policy.md",
        "product-overview.md",
        "security.md",
        "api-docs.md",
        "uptime-sla.md"
    ]
    num_chunks = rag.index(SAMPLE_DOCUMENTS)
    print(f"  Indexed {len(SAMPLE_DOCUMENTS)} documents into {num_chunks} chunks")
    print(f"  Vocabulary size: {len(rag.vocab)} terms")

    queries = [
        "What is the refund policy for enterprise customers?",
        "What are the API rate limits?",
        "How is customer data encrypted?",
        "What happens if uptime falls below the SLA?",
        "How much does the Professional plan cost?"
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        result = rag.query(query, top_k=3)
        print(f"  Answer: {result['answer']}")
        print(f"  Retrieved {len(result['retrieved'])} chunks:")
        for r in result["retrieved"]:
            preview = r["chunk"][:80].replace("\n", " ")
            print(f"    [{r['source']}] score={r['score']:.4f} | {preview}...")

    print("\n" + "=" * 60)
    print("STEP 5: Chunk Size Comparison")
    print("=" * 60)

    test_query = "What is the refund policy for enterprise customers?"
    for chunk_size in [20, 50, 100, 200]:
        rag_test = RAGPipeline(chunk_size=chunk_size, overlap=max(5, chunk_size // 5))
        n = rag_test.index(SAMPLE_DOCUMENTS)
        result = rag_test.query(test_query, top_k=3)
        top_score = result["retrieved"][0]["score"] if result["retrieved"] else 0
        print(f"  chunk_size={chunk_size:>3d}: {n:>3d} chunks, "
              f"top_score={top_score:.4f}, "
              f"answer_len={len(result['answer'])}")

    print("\n" + "=" * 60)
    print("STEP 6: Prompt Inspection")
    print("=" * 60)

    result = rag.query("What encryption does Acme use?", top_k=2)
    prompt_lines = result["prompt"].split("\n")
    print(f"  Prompt length: {len(result['prompt'])} chars")
    print(f"  Prompt lines: {len(prompt_lines)}")
    print(f"\n  First 5 lines of generated prompt:")
    for line in prompt_lines[:5]:
        print(f"    {line}")
    print(f"  ...")
    print(f"  Last 3 lines of generated prompt:")
    for line in prompt_lines[-3:]:
        print(f"    {line}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  RAG pipeline: Query -> Embed -> Search -> Augment -> Generate")
    print(f"  Documents indexed: {len(SAMPLE_DOCUMENTS)}")
    print(f"  Total chunks: {num_chunks}")
    print(f"  Vocabulary size: {len(rag.vocab)}")
    print(f"  Embedding dimensions: {len(rag.vocab)}")
    print("  Similarity metric: cosine similarity")
    print("  Embedding method: TF-IDF")
    print("\n  In production, replace TF-IDF with neural embeddings")
    print("  (text-embedding-3-small) and the simple generator with")
    print("  an actual LLM API call. The pipeline stays the same.")