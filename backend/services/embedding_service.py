import os
import asyncio
import concurrent.futures
from typing import List
import together
from utils.logger import chat_logger
from config.settings import settings
from utils.cache import cache_service

# Thread pool for concurrent requests
embedding_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # Reduced for Render free tier


class EmbeddingService:
    """Service for generating embeddings using Together.ai API with BAAI/bge-large-en-v1.5 model"""

    @staticmethod
    def get_api_key() -> str:
        """Get Together.ai API key from settings"""
        return settings.TOGETHER_API_KEY

    @staticmethod
    def get_embedding_model() -> str:
        """Get embedding model from settings"""
        return settings.EMBEDDING_MODEL

    @staticmethod
    def get_embedding_dimensions() -> int:
        """Get embedding dimensions from settings"""
        return settings.EMBEDDING_DIMENSIONS

    @staticmethod
    def initialize_client() -> together.Together:
        """Initialize and return Together.ai client"""
        api_key = EmbeddingService.get_api_key()

        chat_logger.debug(
            f"Initializing client with API key: {'[SET]' if api_key else '[NOT SET]'}"
        )
        if not api_key:
            chat_logger.error("TOGETHER_API_KEY is not set in settings")
            raise ValueError("TOGETHER_API_KEY environment variable is required")

        client = together.Together(api_key=api_key)
        return client

    @staticmethod
    async def generate_embedding(text: str, max_retries: int = 3) -> List[float]:
        """
        Generate embedding for a single text using Together.ai API with BAAI/bge-large-en-v1.5
        Uses caching to avoid repeated API calls
        """
        # Check cache first
        cached_embedding = await cache_service.get_cached_embedding(text)
        if cached_embedding:
            return cached_embedding

        loop = asyncio.get_event_loop()
        api_key = EmbeddingService.get_api_key()
        model = EmbeddingService.get_embedding_model()

        if not api_key:
            raise ValueError("Together.ai API key not configured")

        for attempt in range(max_retries):

            def _generate():
                try:
                    client = EmbeddingService.initialize_client()

                    # Truncate text if too long (BAAI model handles up to 512 tokens)
                    # Estimate: ~4 chars per token, so max ~2000 chars
                    text_truncated = text[:2000] if len(text) > 2000 else text

                    chat_logger.debug(f"Generating embedding with model: {model}")

                    response = client.embeddings.create(
                        model=model,
                        input=text_truncated,
                    )
                    return response.data[0].embedding, None
                except Exception as e:
                    return None, e

            try:
                embedding, error = await loop.run_in_executor(embedding_pool, _generate)

                if embedding:
                    # Cache the embedding
                    await cache_service.save_embedding_to_cache(text, embedding)
                    return embedding

                if error:
                    error_str = str(error).lower()

                    # Check if it's a rate limit (temporary)
                    if any(
                        keyword in error_str for keyword in ["rate limit", "429", "503"]
                    ):
                        if attempt < max_retries - 1:
                            wait_time = min(2.0**attempt, 5.0)
                            chat_logger.warning(
                                f"Rate limit hit, waiting {wait_time}s before retry"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            # Max retries reached
                            raise error
                    else:
                        # For other errors, raise immediately
                        raise error

            except Exception as e:
                # For non-embedding errors, raise immediately
                raise

        # If we get here, all retries failed
        raise Exception(f"Failed to generate embedding after {max_retries} attempts")

    @staticmethod
    async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch"""
        tasks = [EmbeddingService.generate_embedding(text) for text in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        result = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, Exception):
                chat_logger.error(
                    f"Failed to generate embedding for text {i}", error=str(emb)
                )
                raise emb
            result.append(emb)

        return result

    @staticmethod
    async def generate_batch_query_embeddings(queries: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple queries in batch"""
        return await EmbeddingService.generate_embeddings_batch(queries)

    @staticmethod
    async def generate_query_embedding(query: str, max_retries: int = 3) -> List[float]:
        """
        Generate embedding for a query text using Together.ai API with BAAI/bge-large-en-v1.5
        """
        loop = asyncio.get_event_loop()
        api_key = EmbeddingService.get_api_key()
        model = EmbeddingService.get_embedding_model()

        if not api_key:
            raise ValueError("Together.ai API key not configured")

        for attempt in range(max_retries):

            def _generate():
                try:
                    client = EmbeddingService.initialize_client()

                    # Truncate query if too long (BAAI model handles up to 512 tokens)
                    # Estimate: ~4 chars per token, so max ~2000 chars
                    query_truncated = query[:2000] if len(query) > 2000 else query

                    chat_logger.debug(f"Generating query embedding with model: {model}")

                    response = client.embeddings.create(
                        model=model,
                        input=query_truncated,
                    )
                    return response.data[0].embedding, None
                except Exception as e:
                    return None, e

            try:
                embedding, error = await loop.run_in_executor(embedding_pool, _generate)

                if embedding:
                    return embedding

                if error:
                    error_str = str(error).lower()

                    # Check if it's a rate limit (temporary)
                    if any(
                        keyword in error_str for keyword in ["rate limit", "429", "503"]
                    ):
                        if attempt < max_retries - 1:
                            wait_time = min(2.0**attempt, 5.0)
                            chat_logger.warning(
                                f"Rate limit hit, waiting {wait_time}s before retry"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            # Max retries reached
                            raise error
                    else:
                        # For other errors, raise immediately
                        raise error

            except Exception as e:
                # For non-embedding errors, raise immediately
                raise

        # If we get here, all retries failed
        raise Exception(
            f"Failed to generate query embedding after {max_retries} attempts"
        )
