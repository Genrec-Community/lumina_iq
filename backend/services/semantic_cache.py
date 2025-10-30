import asyncio
import hashlib
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from config.settings import settings
from utils.logger import get_logger
from utils.cache import RedisCache  # Will be added to utils/cache.py

logger = get_logger("semantic_cache")


class SemanticCache:
    """
    Semantic caching service for embeddings and retrieval results using cosine similarity.
    Provides intelligent caching based on query similarity rather than exact matches.
    """

    def __init__(self):
        self.redis_cache = RedisCache()
        self.similarity_threshold = 0.85  # Cosine similarity threshold
        self.max_candidates = 10  # Maximum candidates to check for similarity

    async def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a_norm = np.array(a) / np.linalg.norm(a)
            b_norm = np.array(b) / np.linalg.norm(b)
            return float(np.dot(a_norm, b_norm))
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _generate_embedding_key(self, query: str, model: str) -> str:
        """Generate cache key for query embeddings."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"embed:{model}:{query_hash}"

    def _generate_retrieval_key(self, token: str, filename: str, query_hash: str) -> str:
        """Generate cache key for retrieval results."""
        return f"retrieval:{token}:{filename}:{query_hash}"

    async def _find_similar_queries(
        self,
        query_embedding: List[float],
        model: str,
        token: Optional[str] = None,
        filename: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Find similar queries in cache based on embedding similarity.

        Returns:
            List of tuples (cache_key, similarity_score) ordered by similarity desc.
        """
        try:
            # Get all embedding keys for this model (this is simplified - in production,
            # you might want to use Redis SCAN or maintain an index)
            pattern = f"embed:{model}:*"
            embedding_keys = await self.redis_cache.scan_keys(pattern)

            candidates = []
            for key in embedding_keys[:self.max_candidates]:  # Limit for performance
                try:
                    cached_data = await self.redis_cache.get(key)
                    if cached_data and isinstance(cached_data, dict):
                        cached_embedding = cached_data.get("embedding")
                        if cached_embedding:
                            similarity = await self._cosine_similarity(
                                query_embedding, cached_embedding
                            )
                            if similarity >= self.similarity_threshold:
                                candidates.append((key, similarity))
                except Exception as e:
                    logger.warning(f"Error checking candidate {key}: {str(e)}")
                    continue

            # Sort by similarity descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates

        except Exception as e:
            logger.error(f"Error finding similar queries: {str(e)}")
            return []

    async def get_cached_embedding(
        self, query: str, model: str
    ) -> Optional[List[float]]:
        """
        Get cached embedding for a query.
        Uses exact match first, then falls back to semantic similarity.
        """
        try:
            # Try exact match first
            cache_key = self._generate_embedding_key(query, model)
            cached_data = await self.redis_cache.get(cache_key)

            if cached_data and isinstance(cached_data, dict):
                embedding = cached_data.get("embedding")
                if embedding:
                    logger.debug(f"Exact embedding cache hit: {model}")
                    return embedding

            # Fallback to semantic similarity (disabled for embeddings as per requirements)
            # This would be too slow for embeddings, so we skip semantic matching here
            return None

        except Exception as e:
            logger.error(f"Error getting cached embedding: {str(e)}")
            return None

    async def set_embedding_cache(
        self,
        query: str,
        embedding: List[float],
        model: str,
        ttl_seconds: int = 86400  # 24 hours
    ) -> bool:
        """Cache query embedding."""
        try:
            cache_key = self._generate_embedding_key(query, model)
            cache_data = {
                "query": query,
                "embedding": embedding,
                "model": model,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl_seconds
            }

            success = await self.redis_cache.set(
                cache_key, cache_data, ttl=ttl_seconds
            )

            if success:
                logger.debug(f"Embedding cached: {model}")
            return success

        except Exception as e:
            logger.error(f"Error setting embedding cache: {str(e)}")
            return False

    async def get_cached_retrieval(
        self,
        query: str,
        query_embedding: List[float],
        token: str,
        filename: str,
        model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached retrieval results using semantic similarity.
        """
        try:
            # First try exact match
            query_hash = hashlib.md5(query.encode()).hexdigest()
            cache_key = self._generate_retrieval_key(token, filename, query_hash)
            cached_data = await self.redis_cache.get(cache_key)

            if cached_data and isinstance(cached_data, dict):
                result = cached_data.get("result")
                if result:
                    logger.debug("Exact retrieval cache hit")
                    return result

            # Find similar queries
            similar_queries = await self._find_similar_queries(
                query_embedding, model, token, filename
            )

            if not similar_queries:
                return None

            # Try the most similar query's retrieval result
            best_match_key, similarity = similar_queries[0]
            best_match_data = await self.redis_cache.get(best_match_key)

            if not best_match_data or not isinstance(best_match_data, dict):
                return None

            # Get the retrieval key for this similar query
            similar_query = best_match_data.get("query", "")
            similar_hash = hashlib.md5(similar_query.encode()).hexdigest()
            retrieval_key = self._generate_retrieval_key(token, filename, similar_hash)

            cached_retrieval = await self.redis_cache.get(retrieval_key)
            if cached_retrieval and isinstance(cached_retrieval, dict):
                result = cached_retrieval.get("result")
                if result:
                    logger.debug(f"Semantic retrieval cache hit (similarity: {similarity:.3f})")
                    return result

            return None

        except Exception as e:
            logger.error(f"Error getting cached retrieval: {str(e)}")
            return None

    async def set_retrieval_cache(
        self,
        query: str,
        query_embedding: List[float],
        result: Dict[str, Any],
        token: str,
        filename: str,
        model: str,
        ttl_seconds: int = 3600  # 1 hour
    ) -> bool:
        """Cache retrieval results."""
        try:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            cache_key = self._generate_retrieval_key(token, filename, query_hash)

            cache_data = {
                "query": query,
                "query_embedding": query_embedding,
                "result": result,
                "token": token,
                "filename": filename,
                "model": model,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl_seconds
            }

            success = await self.redis_cache.set(
                cache_key, cache_data, ttl=ttl_seconds
            )

            if success:
                logger.debug("Retrieval result cached")
            return success

        except Exception as e:
            logger.error(f"Error setting retrieval cache: {str(e)}")
            return False

    async def invalidate_user_cache(self, token: str) -> int:
        """Invalidate all cached data for a specific user."""
        try:
            pattern = f"retrieval:{token}:*"
            keys = await self.redis_cache.scan_keys(pattern)
            deleted_count = 0

            for key in keys:
                if await self.redis_cache.delete(key):
                    deleted_count += 1

            logger.info(f"Invalidated {deleted_count} cache entries for user {token}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error invalidating user cache: {str(e)}")
            return 0

    async def invalidate_document_cache(self, token: str, filename: str) -> int:
        """Invalidate cached data for a specific document."""
        try:
            pattern = f"retrieval:{token}:{filename}:*"
            keys = await self.redis_cache.scan_keys(pattern)
            deleted_count = 0

            for key in keys:
                if await self.redis_cache.delete(key):
                    deleted_count += 1

            logger.info(f"Invalidated {deleted_count} cache entries for document {filename}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error invalidating document cache: {str(e)}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics."""
        try:
            # This is a simplified stats implementation
            # In production, you might want more detailed metrics
            embedding_pattern = "embed:*:*"
            retrieval_pattern = "retrieval:*:*:*"

            embedding_keys = await self.redis_cache.scan_keys(embedding_pattern)
            retrieval_keys = await self.redis_cache.scan_keys(retrieval_pattern)

            return {
                "embedding_cache_entries": len(embedding_keys),
                "retrieval_cache_entries": len(retrieval_keys),
                "similarity_threshold": self.similarity_threshold,
                "max_candidates": self.max_candidates
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}


# Global semantic cache instance
semantic_cache = SemanticCache()