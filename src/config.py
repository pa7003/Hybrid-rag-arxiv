from dataclasses import dataclass

@dataclass
class RAGConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"
    top_k: int = 5
    alpha: float = 0.6
    max_context_length: int = 2000
