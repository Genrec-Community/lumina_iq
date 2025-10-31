"""
Together AI service using LangChain for LLM and embeddings.
Provides unified interface for Together AI's models.
"""

from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config.settings import settings
from utils.logger import get_logger
import asyncio
from functools import lru_cache

logger = get_logger("together_service")


class TogetherService:
    """Service for interacting with Together AI using LangChain"""

    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._initialized = False

    def initialize(self):
        """Initialize Together AI LLM and embeddings"""
        if self._initialized:
            return

        try:
            # Initialize LLM using LangChain's ChatOpenAI with Together AI
            self._llm = ChatOpenAI(
                model=settings.TOGETHER_MODEL or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=0.7,
                max_tokens=2048,
                openai_api_key=settings.TOGETHER_API_KEY,
                openai_api_base=settings.TOGETHER_BASE_URL,
            )

            # Initialize embeddings using LangChain's OpenAIEmbeddings with Together AI
            self._embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.TOGETHER_API_KEY,
                openai_api_base=settings.TOGETHER_BASE_URL,
            )

            self._initialized = True
            logger.info(
                "Together AI service initialized successfully",
                extra={
                    "extra_fields": {
                        "llm_model": settings.TOGETHER_MODEL or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                        "embedding_model": settings.EMBEDDING_MODEL,
                    }
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize Together AI service: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    def get_llm(self) -> ChatOpenAI:
        """Get the initialized LLM instance"""
        if not self._initialized:
            self.initialize()
        return self._llm

    def get_embeddings(self) -> OpenAIEmbeddings:
        """Get the initialized embeddings instance"""
        if not self._initialized:
            self.initialize()
        return self._embeddings

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate text using Together AI LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        if not self._initialized:
            self.initialize()

        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            # Create a new LLM instance with specific parameters
            llm = ChatOpenAI(
                model=settings.TOGETHER_MODEL or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=settings.TOGETHER_API_KEY,
                openai_api_base=settings.TOGETHER_BASE_URL,
            )

            response = await llm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(
                f"Text generation failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Generate text with streaming response.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they are generated
        """
        if not self._initialized:
            self.initialize()

        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            llm = ChatOpenAI(
                model=settings.TOGETHER_MODEL or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=settings.TOGETHER_API_KEY,
                openai_api_base=settings.TOGETHER_BASE_URL,
                streaming=True,
            )

            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(
                f"Streaming generation failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self._initialized:
            self.initialize()

        try:
            embedding = await self._embeddings.aembed_query(text)
            return embedding
        except Exception as e:
            logger.error(
                f"Text embedding failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            self.initialize()

        try:
            # Process in batches to avoid rate limits
            batch_size = settings.EMBEDDING_BATCH_SIZE
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = await self._embeddings.aembed_documents(batch)
                all_embeddings.extend(embeddings)

            return all_embeddings
        except Exception as e:
            logger.error(
                f"Document embedding failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Have a chat conversation with the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            AI response text
        """
        if not self._initialized:
            self.initialize()

        try:
            langchain_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:
                    langchain_messages.append(HumanMessage(content=content))

            llm = ChatOpenAI(
                model=settings.TOGETHER_MODEL or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=settings.TOGETHER_API_KEY,
                openai_api_base=settings.TOGETHER_BASE_URL,
            )

            response = await llm.ainvoke(langchain_messages)
            return response.content

        except Exception as e:
            logger.error(
                f"Chat completion failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise


# Global instance
together_service = TogetherService()
