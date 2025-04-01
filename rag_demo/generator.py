from transformers import pipeline

class Generator:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base")
        self.max_context_chars = 1500  # truncate context to stay under token limit

    def generate(self, context, question):
        # Limit context length to avoid exceeding 512 token limit
        context = context[:self.max_context_chars]

        prompt = f"""
You are an expert assistant answering questions using only the provided context.

Context:
{context}

Question:
{question}

Answer:
""".strip()

        result = self.generator(prompt, max_new_tokens=150)
        return result[0]['generated_text']
