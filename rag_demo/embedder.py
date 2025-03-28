from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def encode_query(self, query):
        return self.encode([query])
