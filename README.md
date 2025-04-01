---
title: Simple RAG Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
---

# Simple RAG Demo

> Retrieval-Augmented Generation using FAISS, Sentence Transformers, and FLAN-T5.

This is a minimal RAG (Retrieval-Augmented Generation) system built with:

- FAISS for efficient document retrieval
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- `google/flan-t5-base` (or `sshleifer/tiny-t5` for lighter use) for answer generation
- Gradio for the user interface

---

## 🚀 How to Run Locally

```bash
poetry install
poetry run python app.py
```

Make sure you have internet to download Hugging Face models the first time.

---

## 🧪 Try Asking:

- "Tell me about Datatonic?"
- "Who are the Datatonic officers?"
- "Where are Datatonic offices?"

---

## 🧠 Dataset

Located in the `/docs` folder as plain text files.

You can add your own `.txt` files to expand the knowledge base!

---

## 🛠 Tech Stack

- Python
- FAISS
- SentenceTransformers
- Hugging Face Transformers
- Gradio
- Poetry

---

Built for fast prototyping and demo purposes.