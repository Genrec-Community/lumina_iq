import os
import asyncio
import concurrent.futures
from typing import List, Optional
import together
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.logger import chat_logger
from config.settings import settings
from utils.cache import cache_service

# Thread pool for concurrent requests
embedding_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Increased for better performance


class EmbeddingService:
    """Service for generating embeddings using Together.ai API with BAAI/bge-large-en-v1.5 model"""

    def __init__(self):
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5  # Open circuit after 5 failures
        self._circuit_breaker_timeout = 60  # 60 seconds timeout
        self._last_failure_time = None
        self._batch_size = getattr(settings, 'EMBEDDING_BATCH_SIZE', 32)

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            if self._last_failure_time:
                import time
                if time.time() - self._last_failure_time < self._circuit_breaker_timeout:
                    return True
                else:
                    # Reset circuit breaker after timeout
                    self._circuit_breaker_failures = 0
                    self._last_failure_time = None
        return False

    def _record_failure(self):
        """Record a failure for circuit breaker"""
        import time
        self._circuit_breaker_failures += 1
        self._last_failure_time = time.time()

    def _record_success(self):
        """Record a success for circuit breaker"""
        self._circuit_breaker_failures = 0
        self._last_failure_time = None

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Generate embedding for a single text using Together.ai API with BAAI/bge-large-en-v1.5
        Uses caching and circuit breaker pattern for resilience
        """
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            chat_logger.warning("Circuit breaker is open, falling back to cached embeddings")
            cached_embedding = await cache_service.get_cached_embedding(text)
            if cached_embedding:
                return cached_embedding
            else:
                raise Exception("Circuit breaker open and no cached embedding available")

        # Check cache first
        cached_embedding = await cache_service.get_cached_embedding(text)
        if cached_embedding:
            self._record_success()
            return cached_embedding

        loop = asyncio.get_event_loop()
        api_key = self.get_api_key()
        model = self.get_embedding_model()

        if not api_key:
            raise ValueError("Together.ai API key not configured")

        for attempt in range(max_retries):

            def _generate():
                try:
                    client = self.initialize_client()

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
                    self._record_success()
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
                            self._record_failure()
                            raise error
                    else:
                        # For other errors, raise immediately
                        self._record_failure()
                        raise error

            except Exception as e:
                # For non-embedding errors, raise immediately
                self._record_failure()
                raise

        # If we get here, all retries failed
        self._record_failure()
        raise Exception(f"Failed to generate embedding after {max_retries} attempts")

    async def generate_embeddings_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in optimized batches
        Uses semaphore for concurrency control and circuit breaker pattern
        """
        if not texts:
            return []

        if batch_size is None:
            batch_size = self._batch_size

        chat_logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")

        # Check circuit breaker
        if self._is_circuit_breaker_open():
            chat_logger.warning("Circuit breaker is open, falling back to cached embeddings")
            cached_embeddings = []
            for text in texts:
                cached = await cache_service.get_cached_embedding(text)
                if cached:
                    cached_embeddings.append(cached)
                else:
                    raise Exception("Circuit breaker open and not all embeddings cached")

            if len(cached_embeddings) == len(texts):
                return cached_embeddings
            else:
                raise Exception("Circuit breaker open and incomplete cached embeddings")

        semaphore = asyncio.Semaphore(batch_size)  # Limit concurrent requests

        async def generate_with_semaphore(text: str) -> List[float]:
            async with semaphore:
                return await self.generate_embedding(text)

        # Process in batches to avoid overwhelming the API
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            chat_logger.debug(f"Processing batch {i//batch_size + 1} with {len(batch_texts)} texts")

            tasks = [generate_with_semaphore(text) for text in batch_texts]
            try:
                batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)

                for j, emb in enumerate(batch_embeddings):
                    if isinstance(emb, Exception):
                        chat_logger.error(
                            f"Failed to generate embedding for text {i+j}: {str(emb)}"
                        )
                        self._record_failure()
                        raise emb
                    else:
                        self._record_success()

                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                chat_logger.error(f"Batch processing failed: {str(e)}")
                raise

        chat_logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings

    async def generate_batch_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple queries in batch"""
        return await self.generate_embeddings_batch(queries)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_query_embedding(self, query: str, max_retries: int = 3) -> List[float]:
        """
        Generate embedding for a query text using Together.ai API with BAAI/bge-large-en-v1.5
        """
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            chat_logger.warning("Circuit breaker is open, falling back to cached embeddings")
            cached_embedding = await cache_service.get_cached_embedding(query)
            if cached_embedding:
                return cached_embedding
            else:
                raise Exception("Circuit breaker open and no cached embedding available")

        # Check cache first
        cached_embedding = await cache_service.get_cached_embedding(query)
        if cached_embedding:
            self._record_success()
            return cached_embedding

        loop = asyncio.get_event_loop()
        api_key = self.get_api_key()
        model = self.get_embedding_model()

        if not api_key:
            raise ValueError("Together.ai API key not configured")

        for attempt in range(max_retries):

            def _generate():
                try:
                    client = self.initialize_client()

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
                    # Cache the embedding
                    await cache_service.save_embedding_to_cache(query, embedding)
                    self._record_success()
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
                            self._record_failure()
                            raise error
                    else:
                        # For other errors, raise immediately
                        self._record_failure()
                        raise error

            except Exception as e:
                # For non-embedding errors, raise immediately
                self._record_failure()
                raise

        # If we get here, all retries failed
        self._record_failure()
        raise Exception(
            f"Failed to generate query embedding after {max_retries} attempts"
        )


# Global instance for backward compatibility
embedding_service = EmbeddingService()
