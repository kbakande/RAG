from transformers import pipeline

class Generator:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="google/flan-t5-large")

    def generate(self, context, question):
        prompt = f"""You are an expert AI assistant answering questions based on the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:"""

        result = self.generator(prompt, max_new_tokens=150)
        return result[0]['generated_text']
