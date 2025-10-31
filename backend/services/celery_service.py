"""
Celery service for async task management.
"""

from typing import Optional, Dict, Any
from celery import Celery
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("celery_service")


class CeleryService:
    """Service for managing Celery async tasks"""

    def __init__(self):
        self._celery_app: Optional[Celery] = None
        self._initialized = False

    def initialize(self):
        """Initialize Celery application"""
        if self._initialized:
            return

        try:
            # Parse broker and backend URLs with password
            broker_url = f"redis://:{settings.REDIS_URL.split('@')[-1] if '@' in settings.REDIS_URL else settings.CELERY_BROKER_URL}"
            if "8kpszJnpug4WJ1IF2Tv4LShIR4TJfWUU" not in broker_url and "redis-13314" in broker_url:
                broker_url = f"redis://:8kpszJnpug4WJ1IF2Tv4LShIR4TJfWUU@{broker_url.replace('redis://', '')}"

            backend_url = f"redis://:{settings.REDIS_URL.split('@')[-1] if '@' in settings.REDIS_URL else settings.CELERY_RESULT_BACKEND}"
            if "8kpszJnpug4WJ1IF2Tv4LShIR4TJfWUU" not in backend_url and "redis-13314" in backend_url:
                backend_url = f"redis://:8kpszJnpug4WJ1IF2Tv4LShIR4TJfWUU@{backend_url.replace('redis://', '')}"

            self._celery_app = Celery(
                "lumina_iq",
                broker=broker_url,
                backend=backend_url,
            )

            # Configure Celery
            self._celery_app.conf.update(
                task_serializer=settings.CELERY_TASK_SERIALIZER,
                result_serializer=settings.CELERY_RESULT_SERIALIZER,
                accept_content=getattr(settings, 'CELERY_ACCEPT_CONTENT', ['json']),
                timezone=settings.CELERY_TIMEZONE,
                enable_utc=settings.CELERY_ENABLE_UTC,
                task_track_started=True,
                task_time_limit=30 * 60,  # 30 minutes
                task_soft_time_limit=25 * 60,  # 25 minutes
            )

            self._initialized = True
            logger.info("Celery service initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize Celery service: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            # Don't raise - allow application to continue without Celery
            self._initialized = False

    def get_celery_app(self) -> Optional[Celery]:
        """Get the Celery application instance"""
        if not self._initialized:
            self.initialize()
        return self._celery_app

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a Celery task"""
        if not self._initialized:
            return {"status": "celery_not_initialized"}

        try:
            result = self._celery_app.AsyncResult(task_id)
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.ready() else None,
            }
        except Exception as e:
            logger.error(f"Failed to get task status: {str(e)}")
            return {"status": "error", "error": str(e)}


# Global instance
celery_service = CeleryService()
