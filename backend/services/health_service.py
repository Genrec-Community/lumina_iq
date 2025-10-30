"""
Health service for comprehensive monitoring and health checks.
Provides dependency health checks, metrics collection, and status monitoring.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import lru_cache

from config.settings import settings
from utils.logger import get_logger
from services.cache_service import cache_service
from services.celery_service import celery_service
from services.qdrant_service import qdrant_service

logger = get_logger("health_service")


class HealthService:
    """
    Comprehensive health monitoring service for production deployments.
    Handles dependency checks, metrics collection, and circuit breaker status.
    """

    def __init__(self):
        self._health_cache = {}
        self._cache_ttl = 30  # Cache health checks for 30 seconds
        self._metrics_cache = {}
        self._metrics_ttl = 60  # Cache metrics for 60 seconds

        # Circuit breaker states
        self._circuit_breakers = {
            "redis": {"failures": 0, "last_failure": None, "state": "closed"},
            "qdrant": {"failures": 0, "last_failure": None, "state": "closed"},
            "celery": {"failures": 0, "last_failure": None, "state": "closed"},
        }

        # Performance metrics
        self._response_times = []
        self._error_counts = {"4xx": 0, "5xx": 0}
        self._last_reset = datetime.now()

    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health via cache service."""
        try:
            cache_health = await cache_service.get_cache_health()
            redis_available = cache_health.get("redis_available", False)
            latency = cache_health.get("redis_latency_ms")

            return {
                "status": "healthy" if redis_available else "unhealthy",
                "available": redis_available,
                "latency_ms": latency,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Redis health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _check_qdrant_health(self) -> Dict[str, Any]:
        """Check Qdrant health."""
        try:
            health = await qdrant_service.health_check()
            available = health.get("available", False)
            latency = health.get("latency_ms")

            return {
                "status": "healthy" if available else "unhealthy",
                "available": available,
                "latency_ms": latency,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _check_celery_health(self) -> Dict[str, Any]:
        """Check Celery health."""
        try:
            celery_health = celery_service.health_check()
            available = celery_health.get("celery_available", False)
            workers = celery_health.get("active_workers", [])
            queues = celery_health.get("queue_lengths", {})

            return {
                "status": "healthy" if available else "unhealthy",
                "available": available,
                "active_workers": len(workers),
                "worker_names": workers,
                "queue_lengths": queues,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Celery health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _update_circuit_breaker(self, service_name: str, is_healthy: bool):
        """Update circuit breaker state for a service."""
        cb = self._circuit_breakers[service_name]

        if is_healthy:
            cb["failures"] = 0
            cb["state"] = "closed"
        else:
            cb["failures"] += 1
            cb["last_failure"] = datetime.now()

            # Open circuit after 5 consecutive failures
            if cb["failures"] >= 5:
                cb["state"] = "open"
            elif cb["failures"] >= 3:
                cb["state"] = "half-open"

    async def check_liveness(self) -> Dict[str, Any]:
        """Liveness probe - simple check if service is running."""
        return {
            "status": "alive",
            "service": "learning-app-api",
            "timestamp": datetime.now().isoformat(),
            "uptime": self._get_uptime()
        }

    async def check_readiness(self, use_cache: bool = True) -> Dict[str, Any]:
        """Readiness probe - check if service is ready to serve requests."""
        cache_key = "readiness_check"
        now = datetime.now()

        # Check cache if enabled
        if use_cache and cache_key in self._health_cache:
            cached = self._health_cache[cache_key]
            if (now - datetime.fromisoformat(cached["timestamp"])).seconds < self._cache_ttl:
                return cached

        try:
            # Perform all dependency checks concurrently
            redis_health, qdrant_health, celery_health = await asyncio.gather(
                self._check_redis_health(),
                self._check_qdrant_health(),
                self._check_celery_health()
            )

            # Update circuit breakers
            self._update_circuit_breaker("redis", redis_health["status"] == "healthy")
            self._update_circuit_breaker("qdrant", qdrant_health["status"] == "healthy")
            self._update_circuit_breaker("celery", celery_health["status"] == "healthy")

            all_healthy = all([
                redis_health["status"] == "healthy",
                qdrant_health["status"] == "healthy",
                celery_health["status"] == "healthy"
            ])

            result = {
                "status": "ready" if all_healthy else "degraded",
                "service": "learning-app-api",
                "overall_health": all_healthy,
                "checks": {
                    "redis": redis_health,
                    "qdrant": qdrant_health,
                    "celery": celery_health,
                },
                "circuit_breakers": self._circuit_breakers.copy(),
                "timestamp": now.isoformat()
            }

            # Cache the result
            self._health_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Readiness check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "learning-app-api",
                "error": str(e),
                "timestamp": now.isoformat()
            }

    async def get_detailed_health(self) -> Dict[str, Any]:
        """Comprehensive health check with detailed service information."""
        try:
            readiness = await self.check_readiness(use_cache=False)
            cache_stats = await cache_service.get_cache_stats()
            celery_stats = celery_service.get_worker_stats()

            # Get additional metrics
            metrics = await self._collect_metrics()

            return {
                "status": readiness["status"],
                "service": "learning-app-api",
                "readiness": readiness,
                "cache": cache_stats,
                "celery": celery_stats.get("stats", {}),
                "metrics": metrics,
                "system_info": self._get_system_info(),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Detailed health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "learning-app-api",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output."""
        try:
            cache_key = "prometheus_metrics"
            now = datetime.now()

            # Check cache
            if cache_key in self._metrics_cache:
                cached = self._metrics_cache[cache_key]
                if (now - datetime.fromisoformat(cached["timestamp"])).seconds < self._metrics_ttl:
                    return cached["metrics"]

            # Collect health data
            readiness = await self.check_readiness(use_cache=True)
            cache_stats = await cache_service.get_cache_stats()
            celery_stats = celery_service.get_worker_stats()

            metrics_lines = [
                "# HELP lumina_api_health_status Overall API health status (1=healthy, 0=unhealthy)",
                "# TYPE lumina_api_health_status gauge"
            ]

            # Overall health
            health_value = 1 if readiness["status"] == "ready" else 0
            metrics_lines.append(f'lumina_api_health_status {health_value}')

            # Service health metrics
            services = ["redis", "qdrant", "celery"]
            for service in services:
                service_health = readiness["checks"].get(service, {})
                status_value = 1 if service_health.get("status") == "healthy" else 0
                metrics_lines.extend([
                    f'# HELP lumina_api_{service}_health {service} service health status',
                    f'# TYPE lumina_api_{service}_health gauge',
                    f'lumina_api_{service}_health {status_value}'
                ])

                # Latency metrics
                latency = service_health.get("latency_ms")
                if latency is not None:
                    metrics_lines.extend([
                        f'# HELP lumina_api_{service}_latency_ms {service} service latency in milliseconds',
                        f'# TYPE lumina_api_{service}_latency_ms gauge',
                        f'lumina_api_{service}_latency_ms {latency}'
                    ])

            # Cache metrics
            key_counts = cache_stats.get("key_counts", {})
            for cache_type, count in key_counts.items():
                metrics_lines.extend([
                    f'# HELP lumina_api_cache_{cache_type}_count Number of {cache_type} cache entries',
                    f'# TYPE lumina_api_cache_{cache_type}_count gauge',
                    f'lumina_api_cache_{cache_type}_count {count}'
                ])

            # Celery queue metrics
            if "stats" in celery_stats:
                queue_lengths = readiness["checks"].get("celery", {}).get("queue_lengths", {})
                for queue, length in queue_lengths.items():
                    metrics_lines.extend([
                        f'# HELP lumina_api_celery_queue_{queue}_length Length of {queue} queue',
                        f'# TYPE lumina_api_celery_queue_{queue}_length gauge',
                        f'lumina_api_celery_queue_{queue}_length {length}'
                    ])

            # Performance metrics
            if self._response_times:
                avg_response_time = sum(self._response_times) / len(self._response_times)
                metrics_lines.extend([
                    '# HELP lumina_api_avg_response_time_ms Average response time in milliseconds',
                    '# TYPE lumina_api_avg_response_time_ms gauge',
                    f'lumina_api_avg_response_time_ms {avg_response_time}'
                ])

            # Error counts
            for error_type, count in self._error_counts.items():
                metrics_lines.extend([
                    f'# HELP lumina_api_{error_type}_errors Number of {error_type} errors',
                    f'# TYPE lumina_api_{error_type}_errors counter',
                    f'lumina_api_{error_type}_errors {count}'
                ])

            metrics_output = "\n".join(metrics_lines) + "\n"

            # Cache the result
            self._metrics_cache[cache_key] = {
                "metrics": metrics_output,
                "timestamp": now.isoformat()
            }

            return metrics_output

        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {str(e)}")
            return f"# Error generating metrics: {str(e)}\n"

    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect internal performance metrics."""
        now = datetime.now()
        time_since_reset = (now - self._last_reset).total_seconds()

        return {
            "response_times": {
                "average_ms": sum(self._response_times) / len(self._response_times) if self._response_times else 0,
                "count": len(self._response_times),
                "min_ms": min(self._response_times) if self._response_times else 0,
                "max_ms": max(self._response_times) if self._response_times else 0
            },
            "error_counts": self._error_counts.copy(),
            "uptime_seconds": self._get_uptime(),
            "time_since_reset": time_since_reset,
            "circuit_breakers": self._circuit_breakers.copy()
        }

    def _get_uptime(self) -> float:
        """Get service uptime in seconds."""
        # This is a simple implementation - in production you'd track actual start time
        return time.time() - getattr(self, '_start_time', time.time())

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        import platform
        import psutil

        try:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        except ImportError:
            # psutil not available
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version()
            }

    def record_response_time(self, response_time_ms: float):
        """Record a response time for metrics."""
        self._response_times.append(response_time_ms)
        # Keep only last 1000 measurements
        if len(self._response_times) > 1000:
            self._response_times = self._response_times[-1000:]

    def record_error(self, status_code: int):
        """Record an error for metrics."""
        if 400 <= status_code < 500:
            self._error_counts["4xx"] += 1
        elif status_code >= 500:
            self._error_counts["5xx"] += 1

    def reset_metrics(self):
        """Reset performance metrics."""
        self._response_times.clear()
        self._error_counts = {"4xx": 0, "5xx": 0}
        self._last_reset = datetime.now()
        logger.info("Performance metrics reset")


# Global health service instance
health_service = HealthService()