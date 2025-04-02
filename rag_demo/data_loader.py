import logging
import os
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split a given text into overlapping chunks of words.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The number of words per chunk.
        overlap (int): The number of words to overlap between chunks.

    Returns:
        List[str]: A list of chunked text segments.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks


def load_documents(
    doc_dir: str = "docs", chunk_size: int = 80, overlap: int = 20
) -> List[str]:
    """
    Load and chunk all text documents in the specified directory.

    Args:
        doc_dir (str): The path to the folder containing text documents.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between chunks.

    Returns:
        List[str]: A list of all text chunks from all files.
    """
    chunked_docs = []
    try:
        for fname in os.listdir(doc_dir):
            path = os.path.join(doc_dir, fname)
            if os.path.isfile(path) and fname.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunks = split_text(content, chunk_size=chunk_size, overlap=overlap)
                    chunked_docs.extend(chunks)
        logger.info(
            f"Loaded and chunked {len(chunked_docs)} total segments from '{doc_dir}'."
        )
    except Exception as e:
        logger.error(f"Failed to load documents from {doc_dir}: {e}", exc_info=True)
        raise
    return chunked_docs
