"""
Celery tasks for cache maintenance and invalidation operations.
Handles periodic cache cleanup, warming, and optimization.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .celery_app import app
from backend.services.cache_service import cache_service
from backend.services.document_tracking_service import document_tracking_service
from backend.services.semantic_cache import semantic_cache
from utils.logger import get_logger

logger = get_logger("cache_tasks")


@app.task(bind=True, name='tasks.cache_tasks.cleanup_expired_cache')
def cleanup_expired_cache(self) -> Dict[str, Any]:
    """
    Clean up expired cache entries across all cache layers.
    Runs periodically to maintain cache efficiency.

    Returns:
        Dictionary with cleanup statistics
    """
    try:
        logger.info("Starting expired cache cleanup")

        # Get initial cache stats
        initial_stats = asyncio.run(cache_service.get_cache_stats())

        # Clean Redis cache (expired entries are automatically removed by Redis)
        # But we can clear any corrupted or problematic entries
        redis_cleared = asyncio.run(cache_service.redis_cache.clear_expired())

        # Clean file system cache
        file_cleared = cache_service.file_cache.clear_expired_entries()

        # Clean semantic cache expired entries
        semantic_cleared = asyncio.run(semantic_cache.cleanup_expired_entries())

        # Get final cache stats
        final_stats = asyncio.run(cache_service.get_cache_stats())

        cleanup_stats = {
            'status': 'completed',
            'redis_entries_cleared': redis_cleared,
            'file_cache_entries_cleared': file_cleared,
            'semantic_cache_entries_cleared': semantic_cleared,
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'cleanup_at': datetime.now().isoformat()
        }

        logger.info(f"Cache cleanup completed: Redis={redis_cleared}, File={file_cleared}, Semantic={semantic_cleared}")
        return cleanup_stats

    except Exception as e:
        logger.error(f"Cache cleanup failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.cache_tasks.invalidate_user_cache')
def invalidate_user_cache(self, user_token: str) -> Dict[str, Any]:
    """
    Invalidate all cached data for a specific user.
    Useful after user data updates or account changes.

    Args:
        user_token: User session token

    Returns:
        Dictionary with invalidation results
    """
    try:
        logger.info(f"Starting cache invalidation for user: {user_token}")

        # Invalidate user data across all cache layers
        semantic_invalidated = asyncio.run(cache_service.invalidate_user_data(user_token))

        # Invalidate API response cache for user
        api_invalidated = asyncio.run(cache_service.invalidate_api_cache())
        # Note: This invalidates all API cache, not just user's
        # In production, you might want to implement user-keyed API cache

        # Invalidate user session cache
        # This is handled automatically by TTL, but we can force clear if needed
        session_invalidated = 0  # Placeholder

        invalidation_stats = {
            'status': 'completed',
            'user_token': user_token,
            'semantic_cache_invalidated': semantic_invalidated,
            'api_cache_invalidated': api_invalidated,
            'session_cache_invalidated': session_invalidated,
            'total_entries_invalidated': semantic_invalidated + api_invalidated + session_invalidated,
            'invalidated_at': datetime.now().isoformat()
        }

        logger.info(f"User cache invalidation completed: {semantic_invalidated} semantic entries invalidated")
        return invalidation_stats

    except Exception as e:
        logger.error(f"User cache invalidation failed for {user_token}: {str(e)}")
        raise


@app.task(bind=True, name='tasks.cache_tasks.warm_embedding_cache')
def warm_embedding_cache(self, user_token: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
    """
    Pre-warm embedding cache for frequently accessed documents.
    Improves performance by caching embeddings proactively.

    Args:
        user_token: Optional user token to warm cache for specific user
        limit: Maximum number of documents to process

    Returns:
        Dictionary with warming statistics
    """
    try:
        logger.info(f"Starting embedding cache warming for user: {user_token or 'all'}")

        # Get documents to warm cache for
        if user_token:
            documents = asyncio.run(document_tracking_service.get_user_documents(user_token))
        else:
            documents = asyncio.run(document_tracking_service.get_all_documents())

        # Limit the number of documents processed
        documents = documents[:limit]

        logger.info(f"Processing {len(documents)} documents for cache warming")

        total_chunks = 0
        cached_embeddings = 0
        errors = 0

        for i, doc in enumerate(documents):
            try:
                # Update progress
                progress = int((i / len(documents)) * 100)
                self.update_state(state='PROGRESS', meta={
                    'current': i + 1,
                    'total': len(documents),
                    'progress': progress,
                    'current_document': doc.get('filename', 'unknown')
                })

                # Skip if document processing failed
                if doc.get('status') != 'processed':
                    continue

                # Get document chunks from vector store (this is a simplified approach)
                # In practice, you might need to retrieve chunks differently
                # For now, we'll just count the chunks
                chunk_count = doc.get('chunk_count', 0)
                total_chunks += chunk_count

                # Note: Actual embedding warming would require retrieving chunks
                # and generating/caching embeddings, which might be too expensive
                # This is a placeholder for the concept

            except Exception as e:
                logger.error(f"Error warming cache for document {doc.get('filename')}: {str(e)}")
                errors += 1

        warming_stats = {
            'status': 'completed',
            'user_token': user_token,
            'documents_processed': len(documents),
            'total_chunks': total_chunks,
            'cached_embeddings': cached_embeddings,
            'errors': errors,
            'warming_at': datetime.now().isoformat()
        }

        logger.info(f"Embedding cache warming completed: {cached_embeddings} embeddings cached")
        return warming_stats

    except Exception as e:
        logger.error(f"Embedding cache warming failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.cache_tasks.optimize_cache_storage')
def optimize_cache_storage(self) -> Dict[str, Any]:
    """
    Optimize cache storage by removing redundant entries and compressing data.
    Performs maintenance operations to improve cache efficiency.

    Returns:
        Dictionary with optimization results
    """
    try:
        logger.info("Starting cache storage optimization")

        # Get initial stats
        initial_stats = asyncio.run(cache_service.get_cache_stats())

        # Clean up duplicate semantic cache entries
        semantic_duplicates = asyncio.run(semantic_cache.remove_duplicates())

        # Optimize Redis memory usage (if supported)
        redis_optimized = asyncio.run(cache_service.redis_cache.optimize_memory())

        # Clean file cache (remove temp files, etc.)
        file_optimized = cache_service.file_cache.optimize_storage()

        # Get final stats
        final_stats = asyncio.run(cache_service.get_cache_stats())

        optimization_stats = {
            'status': 'completed',
            'semantic_duplicates_removed': semantic_duplicates,
            'redis_optimization': redis_optimized,
            'file_cache_optimization': file_optimized,
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'optimized_at': datetime.now().isoformat()
        }

        logger.info(f"Cache optimization completed: duplicates={semantic_duplicates}")
        return optimization_stats

    except Exception as e:
        logger.error(f"Cache optimization failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.cache_tasks.rebuild_semantic_cache')
def rebuild_semantic_cache(self) -> Dict[str, Any]:
    """
    Rebuild semantic cache index for improved similarity matching.
    Useful after algorithm updates or data structure changes.

    Returns:
        Dictionary with rebuild statistics
    """
    try:
        logger.info("Starting semantic cache rebuild")

        # Backup current cache stats
        initial_stats = asyncio.run(semantic_cache.get_cache_stats())

        # Rebuild semantic cache index
        rebuild_result = asyncio.run(semantic_cache.rebuild_index())

        # Get final stats
        final_stats = asyncio.run(semantic_cache.get_cache_stats())

        rebuild_stats = {
            'status': 'completed',
            'initial_stats': initial_stats,
            'rebuild_result': rebuild_result,
            'final_stats': final_stats,
            'rebuilt_at': datetime.now().isoformat()
        }

        logger.info("Semantic cache rebuild completed")
        return rebuild_stats

    except Exception as e:
        logger.error(f"Semantic cache rebuild failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.cache_tasks.cache_health_check')
def cache_health_check(self) -> Dict[str, Any]:
    """
    Perform comprehensive health check on all cache layers.
    Monitors cache performance and identifies issues.

    Returns:
        Dictionary with health check results
    """
    try:
        logger.info("Starting cache health check")

        # Get cache health status
        health = asyncio.run(cache_service.get_cache_health())

        # Test cache operations
        test_results = {}

        # Test Redis connectivity
        try:
            ping_result = asyncio.run(cache_service.redis_cache.ping())
            test_results['redis_ping'] = ping_result
        except Exception as e:
            test_results['redis_ping'] = f"Failed: {str(e)}"

        # Test semantic cache
        try:
            semantic_stats = asyncio.run(semantic_cache.get_cache_stats())
            test_results['semantic_cache'] = 'OK' if semantic_stats else 'Empty'
        except Exception as e:
            test_results['semantic_cache'] = f"Failed: {str(e)}"

        # Test file cache
        try:
            file_health = cache_service.file_cache.health_check()
            test_results['file_cache'] = file_health
        except Exception as e:
            test_results['file_cache'] = f"Failed: {str(e)}"

        # Get performance metrics
        performance = asyncio.run(cache_service.get_cache_stats())

        health_check = {
            'status': 'completed',
            'overall_health': health,
            'test_results': test_results,
            'performance_metrics': performance,
            'checked_at': datetime.now().isoformat()
        }

        # Determine if there are critical issues
        critical_issues = []
        if not health.get('redis_available', False):
            critical_issues.append('Redis unavailable')
        if not health.get('semantic_cache_stats'):
            critical_issues.append('Semantic cache issues')

        if critical_issues:
            health_check['alerts'] = critical_issues
            logger.warning(f"Cache health check found issues: {critical_issues}")

        logger.info("Cache health check completed")
        return health_check

    except Exception as e:
        logger.error(f"Cache health check failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.cache_tasks.cache_usage_report')
def cache_usage_report(self, days: int = 7) -> Dict[str, Any]:
    """
    Generate cache usage report for monitoring and optimization.

    Args:
        days: Number of days to analyze

    Returns:
        Dictionary with usage statistics and recommendations
    """
    try:
        logger.info(f"Generating cache usage report for last {days} days")

        # Get current cache statistics
        current_stats = asyncio.run(cache_service.get_cache_stats())

        # Get historical data (simplified - in practice you'd query logs/metrics)
        # This is a placeholder for actual historical analysis

        # Analyze cache hit rates (simplified)
        hit_rate_analysis = {
            'estimated_hit_rate': 'Unknown',  # Would need actual metrics
            'cache_efficiency': 'Good' if current_stats else 'Unknown'
        }

        # Generate recommendations
        recommendations = []

        if current_stats.get('key_counts', {}).get('embeddings', 0) > 10000:
            recommendations.append("Consider increasing embedding cache TTL")

        if current_stats.get('key_counts', {}).get('api_responses', 0) > 5000:
            recommendations.append("High API cache usage - consider cache optimization")

        # Memory usage analysis
        memory_usage = {
            'redis_memory_mb': 'Unknown',  # Would need Redis INFO command
            'file_cache_size_mb': cache_service.file_cache.get_cache_size_mb()
        }

        usage_report = {
            'status': 'completed',
            'period_days': days,
            'current_stats': current_stats,
            'hit_rate_analysis': hit_rate_analysis,
            'memory_usage': memory_usage,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }

        logger.info("Cache usage report generated")
        return usage_report

    except Exception as e:
        logger.error(f"Cache usage report generation failed: {str(e)}")
        raise