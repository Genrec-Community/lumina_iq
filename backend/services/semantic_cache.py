"""
Semantic caching service for RAG queries.
Caches query results based on semantic similarity.
"""

from typing import Optional, Dict, Any, List
from services.embedding_service import embedding_service
from services.cache_service import cache_service
from utils.logger import get_logger
import json
import hashlib

logger = get_logger("semantic_cache")


class SemanticCache:
    """Semantic caching for RAG queries"""

    def __init__(self):
        self._similarity_threshold = 0.95
        self._query_cache: Dict[str, Dict[str, Any]] = {}

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query if semantically similar query exists.

        Args:
            query: User query

        Returns:
            Cached result or None
        """
        try:
            # First try exact match
            query_hash = self._hash_query(query)
            cached = await cache_service.get(f"semantic_cache:{query_hash}")

            if cached:
                logger.debug(f"Semantic cache hit (exact): {query[:50]}...")
                return cached

            # TODO: Implement semantic similarity matching with embeddings
            # For now, only exact matches work

            return None

        except Exception as e:
            logger.error(f"Semantic cache get failed: {str(e)}")
            return None

    async def set(
        self,
        query: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """
        Cache a query result.

        Args:
            query: User query
            result: Query result to cache
            ttl: Optional TTL in seconds
        """
        try:
            query_hash = self._hash_query(query)

            # Store in cache
            from config.settings import settings
            cache_ttl = ttl or settings.CACHE_TTL_SECONDS

            await cache_service.set(
                f"semantic_cache:{query_hash}",
                result,
                ttl=cache_ttl,
            )

            logger.debug(f"Cached query result: {query[:50]}...")

        except Exception as e:
            logger.error(f"Semantic cache set failed: {str(e)}")

    def _hash_query(self, query: str) -> str:
        """Generate hash for query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    async def clear(self):
        """Clear all semantic cache entries"""
        try:
            await cache_service.clear_pattern("semantic_cache:*")
            logger.info("Semantic cache cleared")
        except Exception as e:
            logger.error(f"Semantic cache clear failed: {str(e)}")


# Global instance
semantic_cache = SemanticCache()
