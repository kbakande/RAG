from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

class Embedder:
    def __init__(self, index_path="embeddings.index", docs_path="docs.json"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = index_path
        self.docs_path = docs_path
        self.index = None
        self.docs = []

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def save_index(self, embeddings, docs):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)
        with open(self.docs_path, "w") as f:
            json.dump(docs, f)

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.docs_path, "r") as f:
                self.docs = json.load(f)
            return True
        return False

    def get_index_and_docs(self):
        return self.index, self.docs
