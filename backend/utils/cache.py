import os
import hashlib
import json
import aiofiles
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from config.settings import settings
from utils.logger import get_logger

# Use enhanced logger
logger = get_logger("cache")
import time


class CacheService:
    """Service for caching extracted PDF text to improve performance"""

    def __init__(self):
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

    def _generate_cache_key(self, file_path: str) -> str:
        """Generate a unique cache key based on file path and modification time"""
        try:
            # Get file stats
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            modification_time = file_stat.st_mtime

            # Create a unique identifier combining path, size, and modification time
            unique_string = f"{file_path}_{file_size}_{modification_time}"

            # Generate MD5 hash for the cache key
            cache_key = hashlib.md5(unique_string.encode()).hexdigest()

            return cache_key
        except Exception as e:
            logger.error(
                "Error generating cache key", file_path=file_path, error=str(e)
            )
            # Fallback to just file path hash if stat fails
            return hashlib.md5(file_path.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the full path to the cache file"""
        return self.cache_dir / f"{cache_key}.json"

    async def get_cached_text(self, file_path: str) -> Optional[str]:
        """
        Retrieve cached extracted text for a PDF file

        Args:
            file_path: Path to the PDF file

        Returns:
            Cached text if available, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(file_path)
            cache_file_path = self._get_cache_file_path(cache_key)

            if not cache_file_path.exists():
                logger.debug("No cache found", file_path=file_path)
                return None

            # Read cached data
            async with aiofiles.open(cache_file_path, "r", encoding="utf-8") as f:
                cache_data = json.loads(await f.read())

            # Verify cache data structure
            if not all(key in cache_data for key in ["text", "cached_at", "file_path"]):
                logger.warning("Invalid cache data structure", file_path=file_path)
                return None

            logger.info(
                "Cache hit", file_path=file_path, cached_at=cache_data["cached_at"]
            )
            return cache_data["text"]

        except Exception as e:
            logger.error("Error reading cache", file_path=file_path, error=str(e))
            return None

    async def save_to_cache(self, file_path: str, extracted_text: str) -> bool:
        """
        Save extracted text to cache

        Args:
            file_path: Path to the PDF file
            extracted_text: The extracted text content

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(file_path)
            cache_file_path = self._get_cache_file_path(cache_key)

            # Prepare cache data
            cache_data = {
                "text": extracted_text,
                "file_path": file_path,
                "cached_at": datetime.now().isoformat(),
                "text_length": len(extracted_text),
                "cache_key": cache_key,
            }

            # Save to cache file
            async with aiofiles.open(cache_file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(cache_data, ensure_ascii=False, indent=2))

            logger.info(
                "Successfully cached text", file_path=file_path, cache_key=cache_key
            )
            return True

        except Exception as e:
            logger.error("Error saving to cache", file_path=file_path, error=str(e))
            return False

    def clear_cache(self) -> int:
        """
        Clear all cached files

        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                deleted_count += 1

            logger.info("Cache cleared", deleted_count=deleted_count)
            return deleted_count

        except Exception as e:
            logger.error("Error clearing cache", error=str(e))
            return 0

    def get_cache_info(self) -> dict:
        """
        Get information about the current cache

        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_files = len(cache_files)
            total_size = sum(f.stat().st_size for f in cache_files)

            return {
                "cache_directory": str(self.cache_dir),
                "total_cached_files": total_files,
                "total_cache_size_bytes": total_size,
                "total_cache_size_mb": round(total_size / (1024 * 1024), 2),
            }

        except Exception as e:
            logger.error("Error getting cache info", error=str(e))
            return {}

    def _generate_embedding_cache_key(self, text: str) -> str:
        """Generate cache key for embedding based on text hash"""
        return hashlib.md5(text.encode()).hexdigest()

    async def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        if not getattr(settings, "CACHE_EMBEDDINGS", False):
            return None

        try:
            cache_key = self._generate_embedding_cache_key(text)
            cache_file_path = self.cache_dir / f"emb_{cache_key}.json"

            if not cache_file_path.exists():
                return None

            async with aiofiles.open(cache_file_path, "r") as f:
                cache_data = json.loads(await f.read())

            # Check TTL
            cached_at = datetime.fromisoformat(cache_data["cached_at"])
            ttl_seconds = getattr(settings, "CACHE_TTL_SECONDS", 3600)
            if datetime.now() - cached_at > timedelta(seconds=ttl_seconds):
                cache_file_path.unlink()  # Remove expired cache
                return None

            logger.debug("Embedding cache hit", cache_key=cache_key[:8])
            return cache_data["embedding"]

        except Exception as e:
            logger.error("Error reading embedding cache", error=str(e))
            return None

    async def save_embedding_to_cache(self, text: str, embedding: List[float]) -> bool:
        """Save embedding to cache"""
        if not getattr(settings, "CACHE_EMBEDDINGS", False):
            return False

        try:
            cache_key = self._generate_embedding_cache_key(text)
            cache_file_path = self.cache_dir / f"emb_{cache_key}.json"

            cache_data = {
                "text": text[:100],  # Store first 100 chars for debugging
                "embedding": embedding,
                "cached_at": datetime.now().isoformat(),
                "text_length": len(text),
            }

            async with aiofiles.open(cache_file_path, "w") as f:
                await f.write(json.dumps(cache_data))

            logger.debug("Embedding cached", cache_key=cache_key[:8])
            return True

        except Exception as e:
            logger.error("Error saving embedding to cache", error=str(e))
            return False

    async def get_cached_query_result(
        self, query: str, token: str, filename: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        if not getattr(settings, "CACHE_QUERY_RESULTS", False):
            return None

        try:
            cache_key = hashlib.md5(f"{query}_{token}_{filename}".encode()).hexdigest()
            cache_file_path = self.cache_dir / f"query_{cache_key}.json"

            if not cache_file_path.exists():
                return None

            async with aiofiles.open(cache_file_path, "r") as f:
                cache_data = json.loads(await f.read())

            # Check TTL
            cached_at = datetime.fromisoformat(cache_data["cached_at"])
            ttl_seconds = getattr(settings, "CACHE_TTL_SECONDS", 3600)
            if datetime.now() - cached_at > timedelta(seconds=ttl_seconds):
                cache_file_path.unlink()
                return None

            logger.debug("Query result cache hit", cache_key=cache_key[:8])
            return cache_data["result"]

        except Exception as e:
            logger.error("Error reading query cache", error=str(e))
            return None

    async def save_query_result_to_cache(
        self, query: str, token: str, filename: str, result: Dict[str, Any]
    ) -> bool:
        """Save query result to cache"""
        if not getattr(settings, "CACHE_QUERY_RESULTS", False):
            return False

        try:
            cache_key = hashlib.md5(f"{query}_{token}_{filename}".encode()).hexdigest()
            cache_file_path = self.cache_dir / f"query_{cache_key}.json"

            cache_data = {
                "query": query[:100],
                "token": token,
                "filename": filename,
                "result": result,
                "cached_at": datetime.now().isoformat(),
            }

            async with aiofiles.open(cache_file_path, "w") as f:
                await f.write(json.dumps(cache_data))

            logger.debug("Query result cached", cache_key=cache_key[:8])
            return True

        except Exception as e:
            logger.error("Error saving query result to cache", error=str(e))
            return False


# Global cache service instance
cache_service = CacheService()
