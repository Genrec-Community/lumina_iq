"""
Embedding service with caching support.
Wraps Together AI embeddings with Redis caching.
"""

from typing import List, Optional
import hashlib
import json
from services.together_service import together_service
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("embedding_service")


class EmbeddingService:
    """Service for generating and caching embeddings"""

    def __init__(self):
        self._cache = {}  # Simple in-memory cache
        self._cache_enabled = settings.CACHE_EMBEDDINGS

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query with caching.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []

        # Check cache if enabled
        if self._cache_enabled:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return self._cache[cache_key]

        # Generate embedding
        try:
            embedding = await together_service.embed_text(text)

            # Cache the result
            if self._cache_enabled:
                cache_key = self._get_cache_key(text)
                self._cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(
                f"Failed to generate embedding: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents with caching.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []

        try:
            embeddings = []
            texts_to_embed = []
            indices_to_embed = []

            # Check cache for each text
            for idx, text in enumerate(valid_texts):
                if self._cache_enabled:
                    cache_key = self._get_cache_key(text)
                    if cache_key in self._cache:
                        embeddings.append(self._cache[cache_key])
                    else:
                        texts_to_embed.append(text)
                        indices_to_embed.append(idx)
                else:
                    texts_to_embed.append(text)

            # Generate embeddings for uncached texts
            if texts_to_embed:
                new_embeddings = await together_service.embed_documents(texts_to_embed)

                # Cache new embeddings
                if self._cache_enabled:
                    for text, embedding in zip(texts_to_embed, new_embeddings):
                        cache_key = self._get_cache_key(text)
                        self._cache[cache_key] = embedding

                # Merge with cached embeddings
                if self._cache_enabled:
                    for idx, embedding in zip(indices_to_embed, new_embeddings):
                        embeddings.insert(idx, embedding)
                else:
                    embeddings = new_embeddings

            return embeddings

        except Exception as e:
            logger.error(
                f"Failed to generate document embeddings: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()

    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_size(self) -> int:
        """Get the current cache size"""
        return len(self._cache)


# Global instance
embedding_service = EmbeddingService()
