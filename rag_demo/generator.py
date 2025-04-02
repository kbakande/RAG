from transformers import pipeline

class Generator:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base")
        self.max_context_chars = 1500  # keep input within model limits

    def generate(self, context, question):
        context = context[:self.max_context_chars]

        prompt = f"""
You are a helpful assistant that answers user questions **only** using the context below.

Instructions:
- Give a concise, well-written summary if the context contains relevant info.
- Do NOT copy text verbatim; synthesize and rewrite clearly.
- If the answer is not in the context, reply with: "I don’t know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
""".strip()

        result = self.generator(prompt, max_new_tokens=150)
        return result[0]['generated_text']
