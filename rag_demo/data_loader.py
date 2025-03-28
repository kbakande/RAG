import os

def load_documents(doc_dir="docs"):
    docs, names = [], []
    for fname in os.listdir(doc_dir):
        with open(os.path.join(doc_dir, fname), "r") as f:
            docs.append(f.read())
            names.append(fname)
    return docs, names
