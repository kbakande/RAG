import faiss
import numpy as np

class Retriever:
    def __init__(self, embeddings):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.docs = []

    def add_docs(self, docs):
        self.docs = docs

    def retrieve(self, query_embedding, top_k=2):
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.docs[i] for i in indices[0]]
