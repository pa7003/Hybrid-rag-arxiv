import streamlit as st
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

@st.cache_resource
def load_rag_components():
    df = load_dataset("data/arxiv_ai_papers.csv")
    documents = preprocess_dataframe(df)

    bm25 = BM25Retriever(documents)
    vector = VectorRetriever(documents)
    hybrid = HybridRetriever(bm25, vector, alpha=config.alpha)
    generator = HuggingFaceGenerator(
        model_name=config.llm_model,
        max_length=256
    )
    return hybrid, generator

st.set_page_config(page_title="Hybrid RAG on arXiv AI Papers")

st.title("📚 Hybrid RAG System – arXiv AI Papers")

query = st.text_input("Ask a question about AI research:")

if query:
    with st.spinner("Retrieving and generating answer..."):
        hybrid, generator = load_rag_components()
        results = hybrid.search(query, top_k=config.top_k)
        contexts = [doc for doc, _ in results]
        answer = generator.generate(query, contexts)

    st.subheader("Answer")
    st.write(answer)
