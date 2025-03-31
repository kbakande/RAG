from transformers import pipeline

class Generator:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base")

    def generate(self, context, question):
        prompt = f"""You are an expert AI assistant answering questions strictly based on the provided context. 
If the answer is not in the context, respond with "I don't know based on the given information."

Context:
{context}

Question:
{question}

Answer:"""
        result = self.generator(prompt, max_new_tokens=150)
        return result[0]['generated_text']
