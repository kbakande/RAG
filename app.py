from rag_demo.data_loader import load_documents
from rag_demo.embedder import Embedder
from rag_demo.retriever import Retriever
from rag_demo.generator import Generator
from rag_demo.interface import build_ui
import numpy as np

embedder = Embedder()

# Try to load cached index
if embedder.load_index():
    index, docs = embedder.get_index_and_docs()
    retriever = Retriever(np.array([]))  # Dummy for init
    retriever.index = index
    retriever.add_docs(docs)
else:
    print("📄 Loading and embedding documents afresh...")
    docs = load_documents("docs", chunk_size=80, overlap=20)
    embeddings = embedder.encode(docs)
    embedder.save_index(embeddings, docs)
    retriever = Retriever(embeddings)
    retriever.add_docs(docs)

generator = Generator()

def rag_pipeline(question):
    query_embedding = embedder.encode([question])
    context_chunks = retriever.retrieve(query_embedding)

    print("\n🔍 Retrieved Chunks:")
    for i, chunk in enumerate(context_chunks):
        print(f"[{i+1}] {chunk}\n")

    context = " ".join(context_chunks)
    return generator.generate(context, question)

ui = build_ui(rag_pipeline)
ui.launch()
