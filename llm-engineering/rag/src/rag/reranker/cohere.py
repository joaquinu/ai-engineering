import cohere
from rag.reranker.base import Reranker


class CohereReranker(Reranker):
    def __init__(self, model_name: str = "rerank-v4.0-pro"):
        self.client = cohere.ClientV2()
        self.model_name = model_name

    def rerank(self, query, candidates=None, chunks=None, top_k=None, documents=None):
        if chunks is None:
            # Called with raw documents (not pipeline candidates)
            docs = candidates if documents is None else documents
            results = self.client.rerank(model=self.model_name, query=query, documents=docs, top_k=top_k)
            for result in results:
                print(f"Document: {result.document.text}, Score: {result.score}")
            return results

        doc_ids = [c[0] for c in candidates]
        docs_to_rerank = [chunks[doc_id] for doc_id in doc_ids]
        if not docs_to_rerank:
            return []
        results = self.client.rerank(model=self.model_name, query=query, documents=docs_to_rerank, top_k=len(docs_to_rerank))
        return [(doc_ids[r.index], r.score) for r in results]
