from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class BM25Retriever:
    """
    Keyword-based retriever using BM25.
    """

    def __init__(self, documents: List[str]):
        if not documents:
            raise ValueError("Documents list is empty")

        self.documents = documents
        tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        # Simple improved tokenizer (punctuation removed)
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        top_k = min(top_k, len(self.documents))

        scores = self.bm25.get_scores(self._tokenize(query))
        idx = np.argsort(scores)[::-1][:top_k]

        return [(self.documents[i], scores[i]) for i in idx]
