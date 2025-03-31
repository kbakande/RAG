import os
from typing import List

def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def load_documents(doc_dir="docs", chunk_size=80, overlap=20):
    chunked_docs = []
    for fname in os.listdir(doc_dir):
        with open(os.path.join(doc_dir, fname), "r") as f:
            content = f.read()
            chunks = split_text(content, chunk_size=chunk_size, overlap=overlap)
            chunked_docs.extend(chunks)
    return chunked_docs
