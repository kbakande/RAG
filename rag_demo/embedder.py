import hashlib
import json
import logging
import os
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Handles text embedding using SentenceTransformer and caching using FAISS.
    """

    def __init__(
        self,
        index_path: str = "embeddings.index",
        docs_path: str = "docs.json",
        hash_path: str = "docs.hash",
        doc_dir: str = "docs",
    ) -> None:
        """
        Initialize the Embedder with paths and model setup.

        Args:
            index_path (str): Path to store/load FAISS index.
            docs_path (str): Path to store/load document chunks.
            hash_path (str): Path to store/load document hash.
            doc_dir (str): Directory containing source documents.
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = index_path
        self.docs_path = docs_path
        self.hash_path = hash_path
        self.doc_dir = doc_dir
        self.index: Optional[faiss.Index] = None
        self.docs: List[str] = []

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text chunks into dense vector embeddings.

        Args:
            texts (List[str]): List of text chunks.

        Returns:
            np.ndarray: Embedding matrix of shape (N, D).
        """
        logger.info(f"Encoding {len(texts)} text chunks...")
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_docs_hash(self) -> str:
        """
        Compute an MD5 hash of all document files in the docs folder.

        Returns:
            str: Hash string.
        """
        hash_md5 = hashlib.md5()
        for fname in sorted(os.listdir(self.doc_dir)):
            path = os.path.join(self.doc_dir, fname)
            if os.path.isfile(path) and fname.endswith(".txt"):
                with open(path, "rb") as f:
                    hash_md5.update(f.read())
        return hash_md5.hexdigest()

    def save_index(self, embeddings: np.ndarray, docs: List[str]) -> None:
        """
        Save the FAISS index, document chunks, and their hash to disk.

        Args:
            embeddings (np.ndarray): The dense embeddings.
            docs (List[str]): Corresponding document chunks.
        """
        try:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            faiss.write_index(index, self.index_path)
            logger.info(f"FAISS index saved to: {self.index_path}")

            with open(self.docs_path, "w", encoding="utf-8") as f:
                json.dump(docs, f)
            logger.info(f"Document chunks saved to: {self.docs_path}")

            with open(self.hash_path, "w", encoding="utf-8") as f:
                f.write(self.compute_docs_hash())
            logger.info(f"Hash saved to: {self.hash_path}")

        except Exception as e:
            logger.error("Failed to save index or metadata.", exc_info=True)
            raise

    def load_index(self) -> bool:
        """
        Attempt to load the FAISS index and document cache from disk.

        Returns:
            bool: True if successfully loaded and hash matches, else False.
        """
        try:
            if not (
                os.path.exists(self.index_path)
                and os.path.exists(self.docs_path)
                and os.path.exists(self.hash_path)
            ):
                logger.warning("Index or metadata files missing.")
                return False

            current_hash = self.compute_docs_hash()
            with open(self.hash_path, "r", encoding="utf-8") as f:
                saved_hash = f.read().strip()

            if current_hash != saved_hash:
                logger.warning("Document hash mismatch. Recomputing embeddings.")
                return False

            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, "r", encoding="utf-8") as f:
                self.docs = json.load(f)

            logger.info("FAISS index and documents successfully loaded.")
            return True

        except Exception as e:
            logger.error("Error while loading index or documents.", exc_info=True)
            return False

    def get_index_and_docs(self) -> Tuple[faiss.Index, List[str]]:
        """
        Return the FAISS index and corresponding document chunks.

        Returns:
            Tuple[faiss.Index, List[str]]: The index and docs.
        """
        return self.index, self.docs
