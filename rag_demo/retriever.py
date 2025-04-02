import logging
from typing import List

import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves the most relevant document chunks using FAISS similarity search.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        """
        Initialize the Retriever with given embeddings.

        Args:
            embeddings (np.ndarray): Matrix of embedded document chunks.
        """
        if embeddings.size == 0:
            logger.warning("Retriever initialized with empty embeddings.")
            dim = 384  # Fallback to MiniLM default
        else:
            dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        if embeddings.size != 0:
            self.index.add(embeddings)

        self.docs: List[str] = []
        logger.info(f"Retriever initialized with index dimension: {dim}")

    def add_docs(self, docs: List[str]) -> None:
        """
        Attach document chunks to the retriever.

        Args:
            docs (List[str]): Text chunks corresponding to index vectors.
        """
        self.docs = docs
        logger.info(f"{len(docs)} documents attached to retriever.")

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """
        Retrieve the top-k most relevant document chunks.

        Args:
            query_embedding (np.ndarray): Query vector (shape: 1 x D).
            top_k (int): Number of top results to return.

        Returns:
            List[str]: The top-k retrieved document chunks.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Retrieval will return no results.")
            return []

        try:
            distances, indices = self.index.search(query_embedding, top_k)
            results = [self.docs[i] for i in indices[0] if i < len(self.docs)]
            logger.info(f"Retrieved {len(results)} chunks using FAISS.")
            return results
        except Exception as e:
            logger.error("Retrieval failed.", exc_info=True)
            return []
