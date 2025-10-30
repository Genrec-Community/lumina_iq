import asyncio
import hashlib
import json
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta

from config.settings import settings
from utils.logger import get_logger
from services.semantic_cache import semantic_cache
from utils.cache import RedisCache
from utils.cache import CacheService as FileSystemCache

logger = get_logger("cache_service")


class CacheService:
    """
    Unified caching service that manages both Redis-based semantic caching
    and regular caching for API responses and frequently accessed data.
    """

    def __init__(self):
        self.redis_cache = RedisCache()
        self.file_cache = FileSystemCache()  # Fallback to existing file cache
        self.semantic_cache = semantic_cache

        # Cache TTL configurations
        self.ttl_config = {
            "embeddings": 86400,  # 24 hours
            "retrieval": 3600,    # 1 hour
            "api_responses": 1800,  # 30 minutes
            "user_session": 86400,  # 24 hours
            "metadata": 3600,     # 1 hour
        }

    # ===== SEMANTIC CACHE METHODS =====

    async def get_embedding(
        self, query: str, model: str = None
    ) -> Optional[List[float]]:
        """Get cached embedding for a query."""
        if model is None:
            model = settings.EMBEDDING_MODEL

        return await self.semantic_cache.get_cached_embedding(query, model)

    async def set_embedding(
        self,
        query: str,
        embedding: List[float],
        model: str = None,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache query embedding."""
        if model is None:
            model = settings.EMBEDDING_MODEL

        ttl = ttl_seconds or self.ttl_config["embeddings"]
        return await self.semantic_cache.set_embedding_cache(query, embedding, model, ttl)

    async def get_retrieval_result(
        self,
        query: str,
        query_embedding: List[float],
        token: str,
        filename: str,
        model: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached retrieval results."""
        if model is None:
            model = settings.EMBEDDING_MODEL

        return await self.semantic_cache.get_cached_retrieval(
            query, query_embedding, token, filename, model
        )

    async def set_retrieval_result(
        self,
        query: str,
        query_embedding: List[float],
        result: Dict[str, Any],
        token: str,
        filename: str,
        model: str = None,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache retrieval results."""
        if model is None:
            model = settings.EMBEDDING_MODEL

        ttl = ttl_seconds or self.ttl_config["retrieval"]
        return await self.semantic_cache.set_retrieval_cache(
            query, query_embedding, result, token, filename, model, ttl
        )

    # ===== REGULAR CACHE METHODS =====

    def _generate_api_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for API responses."""
        # Sort params for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"api:{endpoint}:{params_hash}"

    def _generate_session_key(self, session_id: str) -> str:
        """Generate cache key for user sessions."""
        return f"session:{session_id}"

    def _generate_metadata_key(self, resource_type: str, identifier: str) -> str:
        """Generate cache key for metadata."""
        return f"metadata:{resource_type}:{identifier}"

    async def get_api_response(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached API response."""
        try:
            cache_key = self._generate_api_key(endpoint, params)
            cached_data = await self.redis_cache.get(cache_key)

            if cached_data and isinstance(cached_data, dict):
                response = cached_data.get("response")
                if response:
                    logger.debug(f"API cache hit: {endpoint}")
                    return response

            return None

        except Exception as e:
            logger.error(f"Error getting API cache: {str(e)}")
            return None

    async def set_api_response(
        self,
        endpoint: str,
        params: Dict[str, Any],
        response: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache API response."""
        try:
            cache_key = self._generate_api_key(endpoint, params)
            ttl = ttl_seconds or self.ttl_config["api_responses"]

            cache_data = {
                "endpoint": endpoint,
                "params": params,
                "response": response,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl
            }

            success = await self.redis_cache.set(cache_key, cache_data, ttl=ttl)
            if success:
                logger.debug(f"API response cached: {endpoint}")
            return success

        except Exception as e:
            logger.error(f"Error setting API cache: {str(e)}")
            return False

    async def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session."""
        try:
            cache_key = self._generate_session_key(session_id)
            cached_data = await self.redis_cache.get(cache_key)

            if cached_data and isinstance(cached_data, dict):
                session = cached_data.get("session")
                if session:
                    logger.debug(f"Session cache hit: {session_id}")
                    return session

            return None

        except Exception as e:
            logger.error(f"Error getting session cache: {str(e)}")
            return None

    async def set_user_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache user session."""
        try:
            cache_key = self._generate_session_key(session_id)
            ttl = ttl_seconds or self.ttl_config["user_session"]

            cache_data = {
                "session_id": session_id,
                "session": session_data,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl
            }

            success = await self.redis_cache.set(cache_key, cache_data, ttl=ttl)
            if success:
                logger.debug(f"Session cached: {session_id}")
            return success

        except Exception as e:
            logger.error(f"Error setting session cache: {str(e)}")
            return False

    async def get_metadata(
        self, resource_type: str, identifier: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached metadata."""
        try:
            cache_key = self._generate_metadata_key(resource_type, identifier)
            cached_data = await self.redis_cache.get(cache_key)

            if cached_data and isinstance(cached_data, dict):
                metadata = cached_data.get("metadata")
                if metadata:
                    logger.debug(f"Metadata cache hit: {resource_type}:{identifier}")
                    return metadata

            return None

        except Exception as e:
            logger.error(f"Error getting metadata cache: {str(e)}")
            return None

    async def set_metadata(
        self,
        resource_type: str,
        identifier: str,
        metadata: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Cache metadata."""
        try:
            cache_key = self._generate_metadata_key(resource_type, identifier)
            ttl = ttl_seconds or self.ttl_config["metadata"]

            cache_data = {
                "resource_type": resource_type,
                "identifier": identifier,
                "metadata": metadata,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl
            }

            success = await self.redis_cache.set(cache_key, cache_data, ttl=ttl)
            if success:
                logger.debug(f"Metadata cached: {resource_type}:{identifier}")
            return success

        except Exception as e:
            logger.error(f"Error setting metadata cache: {str(e)}")
            return False

    # ===== CACHE INVALIDATION METHODS =====

    async def invalidate_user_data(self, token: str) -> int:
        """Invalidate all cached data for a user."""
        try:
            # Invalidate semantic cache for user
            semantic_invalidated = await self.semantic_cache.invalidate_user_cache(token)

            # Invalidate API cache (this would need pattern matching in Redis)
            # For now, we'll just return the semantic count
            # In production, you might want to maintain user-key mappings

            logger.info(f"Invalidated user cache for {token}: {semantic_invalidated} entries")
            return semantic_invalidated

        except Exception as e:
            logger.error(f"Error invalidating user cache: {str(e)}")
            return 0

    async def invalidate_document_data(self, token: str, filename: str) -> int:
        """Invalidate cached data for a specific document."""
        try:
            # Invalidate semantic cache for document
            semantic_invalidated = await self.semantic_cache.invalidate_document_cache(
                token, filename
            )

            logger.info(f"Invalidated document cache for {filename}: {semantic_invalidated} entries")
            return semantic_invalidated

        except Exception as e:
            logger.error(f"Error invalidating document cache: {str(e)}")
            return 0

    async def invalidate_api_cache(self, endpoint: str = None) -> int:
        """Invalidate API response cache, optionally for specific endpoint."""
        try:
            pattern = f"api:{endpoint or '*'}:*"
            keys = await self.redis_cache.scan_keys(pattern)
            deleted_count = 0

            for key in keys:
                if await self.redis_cache.delete(key):
                    deleted_count += 1

            logger.info(f"Invalidated {deleted_count} API cache entries")
            return deleted_count

        except Exception as e:
            logger.error(f"Error invalidating API cache: {str(e)}")
            return 0

    async def invalidate_metadata_cache(self, resource_type: str = None) -> int:
        """Invalidate metadata cache, optionally for specific resource type."""
        try:
            pattern = f"metadata:{resource_type or '*'}:*"
            keys = await self.redis_cache.scan_keys(pattern)
            deleted_count = 0

            for key in keys:
                if await self.redis_cache.delete(key):
                    deleted_count += 1

            logger.info(f"Invalidated {deleted_count} metadata cache entries")
            return deleted_count

        except Exception as e:
            logger.error(f"Error invalidating metadata cache: {str(e)}")
            return 0

    async def clear_all_cache(self) -> Dict[str, int]:
        """Clear all cache data across all cache types."""
        try:
            results = {}

            # Clear Redis cache
            redis_cleared = await self.redis_cache.clear_all()
            results["redis_keys"] = redis_cleared

            # Clear file system cache (existing functionality)
            file_cleared = self.file_cache.clear_cache()
            results["file_cache_files"] = file_cleared

            logger.info(f"Cleared all caches: {results}")
            return results

        except Exception as e:
            logger.error(f"Error clearing all cache: {str(e)}")
            return {"error": str(e)}

    # ===== MONITORING AND HEALTH CHECKS =====

    async def get_cache_health(self) -> Dict[str, Any]:
        """Get cache system health status."""
        try:
            health = {
                "redis_available": False,
                "redis_latency_ms": None,
                "file_cache_available": True,  # File cache is always available
                "semantic_cache_stats": {},
                "timestamp": datetime.now().isoformat()
            }

            # Check Redis health
            redis_start = datetime.now()
            try:
                await self.redis_cache.ping()
                health["redis_available"] = True
                health["redis_latency_ms"] = (datetime.now() - redis_start).total_seconds() * 1000
            except Exception as e:
                logger.warning(f"Redis health check failed: {str(e)}")

            # Get semantic cache stats
            try:
                health["semantic_cache_stats"] = await self.semantic_cache.get_cache_stats()
            except Exception as e:
                logger.warning(f"Semantic cache stats failed: {str(e)}")

            return health

        except Exception as e:
            logger.error(f"Error getting cache health: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            stats = {
                "health": await self.get_cache_health(),
                "ttl_config": self.ttl_config,
                "timestamp": datetime.now().isoformat()
            }

            # Get key counts by pattern (simplified)
            patterns = {
                "embeddings": "embed:*:*",
                "retrievals": "retrieval:*:*:*",
                "api_responses": "api:*:*",
                "sessions": "session:*",
                "metadata": "metadata:*:*"
            }

            key_counts = {}
            for name, pattern in patterns.items():
                try:
                    keys = await self.redis_cache.scan_keys(pattern)
                    key_counts[name] = len(keys)
                except Exception:
                    key_counts[name] = 0

            stats["key_counts"] = key_counts
            return stats

        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}

    async def initialize(self) -> None:
        """Initialize the cache service and all cache backends."""
        logger.info("Initializing cache service...")

        # Initialize Redis cache (critical for performance)
        try:
            await self.redis_cache.initialize()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {str(e)}")
            logger.warning("Cache service will fallback to file-based caching only")
            # Don't raise - allow fallback to file cache

        # Semantic cache uses lazy initialization, no explicit init needed
        # File cache doesn't need initialization

        # Test cache functionality
        try:
            health = await self.get_cache_health()
            if health.get('redis_available', False):
                logger.info("Cache service initialized successfully with Redis support")
            else:
                logger.warning("Cache service initialized with limited functionality (Redis unavailable)")
        except Exception as e:
            logger.error(f"Error testing cache health during initialization: {str(e)}")
            logger.warning("Cache service initialized but health check failed")

    async def close(self) -> None:
        """Close all cache connections and cleanup resources."""
        try:
            logger.info("Closing cache service...")

            # Redis cache doesn't need explicit closing (connections are managed by redis-py)
            # Semantic cache doesn't need explicit closing
            # File cache doesn't need closing

            logger.info("Cache service closed successfully")

        except Exception as e:
            logger.error(f"Error closing cache service: {str(e)}")
            raise


# Global cache service instance
cache_service = CacheService()