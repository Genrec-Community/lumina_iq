"""
Health check service for monitoring system status.
"""

from typing import Dict, Any
import time
import psutil
from config.settings import settings
from utils.logger import get_logger
from collections import deque

logger = get_logger("health_service")


class HealthService:
    """Service for health checks and system monitoring"""

    def __init__(self):
        self._start_time = time.time()
        self._response_times = deque(maxlen=1000)
        self._error_counts = {"4xx": 0, "5xx": 0}

    def record_response_time(self, response_time_ms: float):
        """Record a response time for metrics"""
        self._response_times.append(response_time_ms)

    def record_error(self, status_code: int):
        """Record an error for metrics"""
        if 400 <= status_code < 500:
            self._error_counts["4xx"] += 1
        elif 500 <= status_code < 600:
            self._error_counts["5xx"] += 1

    async def check_liveness(self) -> Dict[str, Any]:
        """Simple liveness check"""
        return {
            "status": "alive",
            "service": "lumina_iq",
            "timestamp": time.time(),
            "uptime_seconds": int(time.time() - self._start_time),
        }

    async def check_readiness(self) -> Dict[str, Any]:
        """Check if service is ready to handle requests"""
        dependencies = {}
        all_healthy = True

        # Check cache service
        try:
            from services.cache_service import cache_service
            cache_stats = await cache_service.get_stats()
            dependencies["cache"] = {
                "healthy": cache_stats.get("status") == "connected",
                "details": cache_stats,
            }
            if not dependencies["cache"]["healthy"]:
                all_healthy = False
        except Exception as e:
            dependencies["cache"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        # Check Qdrant
        try:
            collection_info = await self.check_qdrant_connection()
            dependencies["qdrant"] = {
                "healthy": collection_info.get("status") == "connected",
                "details": collection_info,
            }
            if not dependencies["qdrant"]["healthy"]:
                all_healthy = False
        except Exception as e:
            dependencies["qdrant"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        # Check AI service
        try:
            ai_status = await self.check_ai_service()
            dependencies["ai_service"] = {
                "healthy": ai_status.get("status") == "operational",
                "details": ai_status,
            }
            if not dependencies["ai_service"]["healthy"]:
                all_healthy = False
        except Exception as e:
            dependencies["ai_service"] = {"healthy": False, "error": str(e)}
            all_healthy = False

        status = "ready" if all_healthy else "degraded"

        return {
            "status": status,
            "service": "lumina_iq",
            "timestamp": time.time(),
            "dependencies": dependencies,
        }

    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Get readiness info
            readiness = await self.check_readiness()

            health_data = {
                "status": readiness["status"],
                "service": "lumina_iq",
                "timestamp": time.time(),
                "environment": settings.ENVIRONMENT,
                "uptime_seconds": int(time.time() - self._start_time),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "disk_percent": disk.percent,
                },
                "dependencies": readiness["dependencies"],
                "metrics": {
                    "response_times": {
                        "count": len(self._response_times),
                        "avg_ms": sum(self._response_times) / len(self._response_times) if self._response_times else 0,
                        "min_ms": min(self._response_times) if self._response_times else 0,
                        "max_ms": max(self._response_times) if self._response_times else 0,
                    },
                    "errors": self._error_counts,
                },
            }

            # Check if system is under heavy load
            if cpu_percent > 90 or memory.percent > 90:
                health_data["status"] = "degraded"
                health_data["warnings"] = []

                if cpu_percent > 90:
                    health_data["warnings"].append("High CPU usage")
                if memory.percent > 90:
                    health_data["warnings"].append("High memory usage")

            return health_data

        except Exception as e:
            logger.error(
                f"Detailed health check failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {
                "status": "unhealthy",
                "service": "lumina_iq",
                "timestamp": time.time(),
                "error": str(e),
            }

    async def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        try:
            health_data = await self.get_detailed_health()
            
            metrics = []
            
            # Health status
            status_value = 1 if health_data["status"] in ["ready", "healthy"] else 0
            metrics.append(f"lumina_iq_health_status {status_value}")
            
            # Uptime
            metrics.append(f"lumina_iq_uptime_seconds {health_data['uptime_seconds']}")
            
            # System metrics
            system = health_data.get("system", {})
            metrics.append(f"lumina_iq_cpu_percent {system.get('cpu_percent', 0)}")
            metrics.append(f"lumina_iq_memory_percent {system.get('memory_percent', 0)}")
            metrics.append(f"lumina_iq_disk_percent {system.get('disk_percent', 0)}")
            
            # Response times
            response_metrics = health_data.get("metrics", {}).get("response_times", {})
            metrics.append(f"lumina_iq_response_time_avg_ms {response_metrics.get('avg_ms', 0)}")
            metrics.append(f"lumina_iq_response_time_max_ms {response_metrics.get('max_ms', 0)}")
            
            # Error counts
            errors = health_data.get("metrics", {}).get("errors", {})
            metrics.append(f"lumina_iq_errors_4xx {errors.get('4xx', 0)}")
            metrics.append(f"lumina_iq_errors_5xx {errors.get('5xx', 0)}")
            
            return "\n".join(metrics) + "\n"
            
        except Exception as e:
            logger.error(f"Prometheus metrics generation failed: {str(e)}")
            return f"# Error generating metrics: {str(e)}\n"

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status (alias for get_detailed_health)"""
        return await self.get_detailed_health()

    @staticmethod
    async def check_qdrant_connection() -> Dict[str, Any]:
        """Check Qdrant vector store connection"""
        try:
            from services.qdrant_service import qdrant_service

            if not qdrant_service._initialized:
                qdrant_service.initialize()

            collection_info = await qdrant_service.get_collection_info()

            return {
                "status": "connected",
                "collection": collection_info,
            }

        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
            }

    @staticmethod
    async def check_cache_connection() -> Dict[str, Any]:
        """Check Redis cache connection"""
        try:
            from services.cache_service import cache_service

            if not cache_service._initialized:
                await cache_service.initialize()

            stats = await cache_service.get_stats()

            return {
                "status": "connected",
                "stats": stats,
            }

        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
            }

    @staticmethod
    async def check_ai_service() -> Dict[str, Any]:
        """Check Together AI service"""
        try:
            from services.together_service import together_service

            if not together_service._initialized:
                together_service.initialize()

            # Try a simple generation
            test_response = await together_service.generate(
                "Say 'OK'",
                max_tokens=10,
            )

            return {
                "status": "operational",
                "test_response_length": len(test_response),
            }

        except Exception as e:
            logger.error(f"AI service health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
            }


# Global instance
health_service = HealthService()
