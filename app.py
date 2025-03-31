from rag_demo.data_loader import load_documents
from rag_demo.embedder import Embedder
from rag_demo.retriever import Retriever
from rag_demo.generator import Generator
from rag_demo.interface import build_ui

# Load and chunk documents
docs = load_documents("docs", chunk_size=80, overlap=20)
embedder = Embedder()
embeddings = embedder.encode(docs)

retriever = Retriever(embeddings)
retriever.add_docs(docs)

generator = Generator()

def rag_pipeline(question):
    query_embedding = embedder.encode_query(question)
    context_chunks = retriever.retrieve(query_embedding)
    context = " ".join(context_chunks)
    return generator.generate(context, question)

ui = build_ui(rag_pipeline)
ui.launch()
