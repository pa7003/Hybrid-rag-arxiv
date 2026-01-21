from src.bm25_retriever import BM25Retriever
from src.vector_retriever import VectorRetriever
from src.hybrid_retriever import HybridRetriever

def test_hybrid_search_returns_results():
    documents = [
        "machine learning is a field of ai",
        "deep learning uses neural networks",
        "bm25 is a keyword search algorithm",
        "faiss supports vector similarity search"
    ]

    bm25 = BM25Retriever(documents)
    vector = VectorRetriever(documents)
    hybrid = HybridRetriever(bm25, vector)

    results = hybrid.search("machine learning", top_k=2)

    assert len(results) > 0
