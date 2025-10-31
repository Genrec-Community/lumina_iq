"""
Cache service using Redis for distributed caching.
Provides caching for embeddings, query results, and document data.
"""

from typing import Optional, Any, Dict
import json
import redis.asyncio as redis
from config.settings import settings
from utils.logger import get_logger
import hashlib

logger = get_logger("cache_service")


class CacheService:
    """Redis-based caching service"""

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._initialized = False

    async def initialize(self):
        """Initialize Redis connection"""
        if self._initialized:
            return

        try:
            # Parse Redis URL
            redis_url = settings.REDIS_URL
            
            # Check if URL includes password
            if "@" in redis_url:
                # Format: redis://password@host:port or redis://host:port
                parts = redis_url.replace("redis://", "").split("@")
                if len(parts) == 2:
                    password = parts[0]
                    host_port = parts[1]
                else:
                    password = None
                    host_port = parts[0]
            else:
                # Check .env for password comment or separate config
                password = "8kpszJnpug4WJ1IF2Tv4LShIR4TJfWUU"  # From .env comment
                host_port = redis_url.replace("redis://", "")
            
            host, port = host_port.split(":")
            
            self._redis = redis.Redis(
                host=host,
                port=int(port),
                password=password,
                db=settings.REDIS_CACHE_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            await self._redis.ping()
            self._initialized = True

            logger.info(
                "Cache service initialized successfully",
                extra={"extra_fields": {"redis_db": settings.REDIS_CACHE_DB}},
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize cache service: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            # Don't raise - allow application to continue without cache
            self._initialized = False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._initialized:
            return None

        try:
            value = await self._redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(
                f"Cache get failed: {str(e)}",
                extra={"extra_fields": {"key": key, "error_type": type(e).__name__}},
            )
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL"""
        if not self._initialized:
            return

        try:
            serialized = json.dumps(value)
            if ttl:
                await self._redis.setex(key, ttl, serialized)
            else:
                await self._redis.set(key, serialized)
        except Exception as e:
            logger.error(
                f"Cache set failed: {str(e)}",
                extra={"extra_fields": {"key": key, "error_type": type(e).__name__}},
            )

    async def delete(self, key: str):
        """Delete value from cache"""
        if not self._initialized:
            return

        try:
            await self._redis.delete(key)
        except Exception as e:
            logger.error(
                f"Cache delete failed: {str(e)}",
                extra={"extra_fields": {"key": key, "error_type": type(e).__name__}},
            )

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._initialized:
            return False

        try:
            return await self._redis.exists(key) > 0
        except Exception as e:
            logger.error(
                f"Cache exists check failed: {str(e)}",
                extra={"extra_fields": {"key": key, "error_type": type(e).__name__}},
            )
            return False

    async def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        if not self._initialized:
            return

        try:
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self._redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache keys matching pattern: {pattern}")
        except Exception as e:
            logger.error(
                f"Cache pattern clear failed: {str(e)}",
                extra={"extra_fields": {"pattern": pattern, "error_type": type(e).__name__}},
            )

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._initialized:
            return {"status": "not_initialized"}

        try:
            info = await self._redis.info()
            return {
                "status": "connected",
                "keys": await self._redis.dbsize(),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            logger.error(
                f"Cache stats retrieval failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._initialized = False
            logger.info("Cache service closed")

    def get_cache_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key with prefix and identifier"""
        return f"{prefix}:{hashlib.md5(identifier.encode()).hexdigest()}"

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information (synchronous fallback)"""
        return {
            "initialized": self._initialized,
            "enabled": settings.CACHE_EMBEDDINGS or settings.CACHE_QUERY_RESULTS,
        }

    def clear_cache(self) -> int:
        """Clear cache (synchronous fallback)"""
        # This is a placeholder for synchronous calls
        logger.warning("Synchronous cache clear called - use async version")
        return 0


# Global instance
cache_service = CacheService()
