from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.logging_config import setup_logging
from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataframe
from src.bm25_retriever import BM25Retriever
from src.vector_retriever import VectorRetriever
from src.hybrid_retriever import HybridRetriever
from src.generator import HuggingFaceGenerator
from src.config import RAGConfig

setup_logging()
config = RAGConfig()

app = FastAPI(title="Hybrid RAG System")

df = load_dataset("data/arxiv_ai.csv")
documents = preprocess_dataframe(df)

bm25 = BM25Retriever(documents)
vector = VectorRetriever(documents)
hybrid = HybridRetriever(bm25, vector, alpha=config.alpha)
generator = HuggingFaceGenerator(
    model_name=config.llm_model,
    max_length=256
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = hybrid.search(req.query, top_k=config.top_k)
    contexts = [doc for doc, _ in results]
    answer = generator.generate(req.query, contexts)

    return {
        "query": req.query,
        "answer": answer
    }
