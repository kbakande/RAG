import logging
from typing import Optional

from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator:
    """
    Generates answers based on retrieved context and a user question
    using a pre-trained generative language model.
    """

    def __init__(
        self, model_name: str = "google/flan-t5-base", max_context_chars: int = 1500
    ) -> None:
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
            context = context[
                : self.max_context_chars
            ]  # Truncate to avoid token overflow

            prompt = f"""
You are a helpful and conversational assistant answering user questions based on provided context.

Instructions:
- Respond in full sentences.
- Use a natural and friendly tone.
- Do not invent information. Only answer using the context.
- If the answer is not available, say "I’m not sure based on the context provided."

Context:
{context}

User Question:
{question}

Response:
""".strip()

            result = self.generator(prompt, max_new_tokens=150)
            answer = result[0]["generated_text"]
            logger.info("Answer successfully generated.")
            return answer

        except Exception as e:
            logger.error("Failed during text generation.", exc_info=True)
            return "Something went wrong while generating the answer."
