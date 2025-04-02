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
        - Give a **clear, concise** summary based on the context.
        - **Do not repeat facts** or phrases — only mention each idea once.
        - If the context does not contain the answer, respond with: "I don’t know based on the provided information."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """.strip()

        result = self.generator(prompt, max_new_tokens=80)
        return result[0]['generated_text']
