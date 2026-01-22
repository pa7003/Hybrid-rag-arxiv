
## Hybrid RAG Pipeline on arXiv AI Research Papers

### Overview
This project implements a production-ready hybrid RAG pipeline using
arXiv AI research papers.

### Features
- BM25 keyword retrieval
- FAISS vector similarity search
- Hybrid ranking
- HuggingFace LLM generation
- Precision@k and Recall@k evaluation
- Modular src structure

### Dataset
https://www.kaggle.com/datasets/yasirabdaali/arxivorg-ai-research-papers-dataset

## Architecture

The system follows a modular RAG architecture:

Query → Hybrid Retrieval (BM25 + FAISS) → Context Fusion → LLM Generation

## Design Decisions

- Hybrid retrieval improves recall and precision
- Score normalization ensures fair combination
- Notebook used for demo, src for production logic

## Error Handling

- Input validation at all entry points
- Fallback strategies for retriever failures
- Structured logging for observability

  ## Future Improvements
- Cross-encoder reranking
- Embedding caching
- Batch processing
- Monitoring and metrics

### How to Run
1. Open the notebook in `notebooks/`
2. Run all cells
3. Modify queries as needed
