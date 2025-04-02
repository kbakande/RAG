import logging
from transformers import pipeline
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Generator:
    """
    Generates answers based on retrieved context and a user question
    using a pre-trained generative language model.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", max_context_chars: int = 1500) -> None:
        """
        Initialize the Generator with a Hugging Face text2text model.

        Args:
            model_name (str): Name of the model to load.
            max_context_chars (int): Max characters of context to include in prompt.
        """
        try:
            self.generator = pipeline("text2text-generation", model=model_name)
            self.max_context_chars = max_context_chars
            logger.info(f"Text generation model '{model_name}' loaded.")
        except Exception as e:
            logger.error("Failed to initialize text generator.", exc_info=True)
            raise

    def generate(self, context: str, question: str) -> str:
        """
        Generate a natural language answer based on context and question.

        Args:
            context (str): Retrieved document chunks.
            question (str): User's natural language question.

        Returns:
            str: The generated answer from the model.
        """
        try:
            context = context[:self.max_context_chars]  # Truncate to avoid token overflow

            prompt = f"""
You are a helpful assistant that answers user questions **only** using the context below.

Instructions:
- Give a concise, well-written summary if the context contains relevant info.
- Do NOT repeat facts or copy text verbatim — synthesize and write clearly.
- If the answer is not in the context, respond with: "I don’t know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
""".strip()

            result = self.generator(prompt, max_new_tokens=150)
            answer = result[0]['generated_text']
            logger.info("Answer successfully generated.")
            return answer

        except Exception as e:
            logger.error("Failed during text generation.", exc_info=True)
            return "Something went wrong while generating the answer."
