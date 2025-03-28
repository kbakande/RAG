from transformers import pipeline

class Generator:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base")

    def generate(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        result = self.generator(prompt, max_new_tokens=100)
        return result[0]['generated_text']
