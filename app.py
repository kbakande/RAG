from rag_demo.data_loader import load_documents
from rag_demo.embedder import Embedder
from rag_demo.retriever import Retriever
from rag_demo.generator import Generator
from rag_demo.interface import build_ui

# Load and embed
docs, _ = load_documents("docs")
embedder = Embedder()
embeddings = embedder.encode(docs)

# Setup retriever
retriever = Retriever(embeddings)
retriever.add_docs(docs)

# Generator
generator = Generator()

# RAG pipeline
def rag_pipeline(question):
    query_embedding = embedder.encode_query(question)
    context_chunks = retriever.retrieve(query_embedding)
    context = " ".join(context_chunks)
    return generator.generate(context, question)

# Gradio interface
ui = build_ui(rag_pipeline)
ui.launch()
