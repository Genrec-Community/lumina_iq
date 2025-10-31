"""
Text generation service using Together AI.
Provides various generation capabilities.
"""

from typing import Optional, List, Dict, Any
from services.together_service import together_service
from utils.logger import get_logger

logger = get_logger("generation_service")


class GenerationService:
    """Service for text generation operations"""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Initialize the generation service"""
        if not self._initialized:
            together_service.initialize()
            self._initialized = True

    async def generate_answer(
        self,
        question: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate an answer to a question given context"""
        if not self._initialized:
            self.initialize()

        system_prompt = """You are a helpful AI assistant. Answer the question based on the provided context.
Be accurate, concise, and helpful. If the context doesn't contain enough information, acknowledge this."""

        prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        try:
            answer = await together_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return answer.strip()

        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "I apologize, but I encountered an error generating the answer."

    async def generate_summary(
        self,
        text: str,
        max_length: int = 200,
    ) -> str:
        """Generate a summary of the text"""
        if not self._initialized:
            self.initialize()

        prompt = f"""Summarize the following text in approximately {max_length} words:

{text[:2000]}

Summary:"""

        try:
            summary = await together_service.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=max_length * 2,
            )
            return summary.strip()

        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return "Failed to generate summary."

    async def generate_explanation(
        self,
        concept: str,
        context: Optional[str] = None,
    ) -> str:
        """Generate an explanation of a concept"""
        if not self._initialized:
            self.initialize()

        if context:
            prompt = f"""Based on this context, explain the concept of '{concept}':

Context:
{context[:1000]}

Provide a clear and comprehensive explanation."""
        else:
            prompt = f"Explain the concept of '{concept}' in a clear and comprehensive way."

        try:
            explanation = await together_service.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=512,
            )
            return explanation.strip()

        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            return "Failed to generate explanation."


# Global instance
generation_service = GenerationService()
