from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import hashlib

class Embedder:
    def __init__(self, index_path="embeddings.index", docs_path="docs.json", hash_path="docs.hash", doc_dir="docs"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = index_path
        self.docs_path = docs_path
        self.hash_path = hash_path
        self.doc_dir = doc_dir
        self.index = None
        self.docs = []

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_docs_hash(self):
        hash_md5 = hashlib.md5()
        for fname in sorted(os.listdir(self.doc_dir)):
            path = os.path.join(self.doc_dir, fname)
            if os.path.isfile(path) and fname.endswith(".txt"):
                with open(path, "rb") as f:
                    hash_md5.update(f.read())
        return hash_md5.hexdigest()

    def save_index(self, embeddings, docs):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)
        with open(self.docs_path, "w") as f:
            json.dump(docs, f)
        with open(self.hash_path, "w") as f:
            f.write(self.compute_docs_hash())

    def load_index(self):
        if not (os.path.exists(self.index_path) and os.path.exists(self.docs_path) and os.path.exists(self.hash_path)):
            return False

        current_hash = self.compute_docs_hash()
        with open(self.hash_path, "r") as f:
            saved_hash = f.read().strip()

        if current_hash != saved_hash:
            return False

        self.index = faiss.read_index(self.index_path)
        with open(self.docs_path, "r") as f:
            self.docs = json.load(f)
        return True

    def get_index_and_docs(self):
        return self.index, self.docs
