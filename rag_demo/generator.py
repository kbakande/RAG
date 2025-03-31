from transformers import pipeline

class Generator:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base")

    def generate(self, context, question):
        prompt = f"""
You are an expert assistant helping answer questions using only the provided context.

Instructions:
- Answer concisely and accurately.
- If the answer is not present in the context, say: "I don’t know based on the given information."

Context:
{context}

Question:
{question}

Answer:
"""
        result = self.generator(prompt.strip(), max_new_tokens=150)
        return result[0]['generated_text']
