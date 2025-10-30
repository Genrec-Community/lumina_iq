"""
Celery configuration and task management service for background operations.
Provides centralized Celery app configuration and task monitoring capabilities.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from celery import Celery
from celery.result import AsyncResult
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("celery_service")


class CeleryService:
    """
    Celery service for managing background tasks with Redis broker and result backend.
    Handles configuration, task registration, and monitoring.
    """

    def __init__(self):
        self.app = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the Celery service."""
        if self._initialized:
            logger.debug("Celery service already initialized")
            return

        try:
            self.app = self._create_celery_app()
            # Configure Celery settings
            self.app.conf.update(
                # Task routing
                task_routes={
                    'tasks.document_tasks.*': {'queue': 'ingestion'},
                    'tasks.cache_tasks.*': {'queue': 'maintenance'},
                    'tasks.rag_tasks.*': {'queue': 'rag_operations'},
                },

                # Task serialization
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                timezone='UTC',
                enable_utc=True,

                # Worker configuration
                worker_prefetch_multiplier=1,  # Process one task at a time
                task_acks_late=True,  # Acknowledge after task completion
                worker_max_tasks_per_child=50,  # Restart worker after 50 tasks

                # Result backend settings
                result_expires=3600,  # Results expire after 1 hour
                result_cache_max=10000,  # Maximum cached results

                # Monitoring and logging
                worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
                worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',

                # Task time limits
                task_soft_time_limit=300,  # 5 minutes soft limit
                task_time_limit=600,  # 10 minutes hard limit

                # Retry configuration
                task_default_retry_delay=60,  # 1 minute retry delay
                task_max_retries=3,

                # Rate limiting
                worker_disable_rate_limits=False,

                # Monitoring
                task_send_sent_event=True,
                task_track_started=True,
            )

            self._initialized = True
            logger.info("Celery service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Celery service: {str(e)}")
            self.app = None
            raise

    def _create_celery_app(self) -> Celery:
        """Create and configure Celery application instance with error handling."""
        try:
            # Redis URLs for broker and result backend
            broker_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/1')
            result_backend = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/1')

            # Create Celery app
            app = Celery(
                'lumina_rag_tasks',
                broker=broker_url,
                backend=result_backend,
                include=[
                    'tasks.document_tasks',
                    'tasks.cache_tasks',
                    'tasks.rag_tasks'
                ]
            )

            logger.info(f"Celery app created successfully with broker: {broker_url}")
            return app

        except Exception as e:
            logger.error(f"Failed to create Celery app: {str(e)}")
            raise

        # Configure Celery settings
        app.conf.update(
            # Task routing
            task_routes={
                'tasks.document_tasks.*': {'queue': 'ingestion'},
                'tasks.cache_tasks.*': {'queue': 'maintenance'},
                'tasks.rag_tasks.*': {'queue': 'rag_operations'},
            },

            # Task serialization
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,

            # Worker configuration
            worker_prefetch_multiplier=1,  # Process one task at a time
            task_acks_late=True,  # Acknowledge after task completion
            worker_max_tasks_per_child=50,  # Restart worker after 50 tasks

            # Result backend settings
            result_expires=3600,  # Results expire after 1 hour
            result_cache_max=10000,  # Maximum cached results

            # Monitoring and logging
            worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
            worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',

            # Task time limits
            task_soft_time_limit=300,  # 5 minutes soft limit
            task_time_limit=600,  # 10 minutes hard limit

            # Retry configuration
            task_default_retry_delay=60,  # 1 minute retry delay
            task_max_retries=3,

            # Rate limiting
            worker_disable_rate_limits=False,

            # Monitoring
            task_send_sent_event=True,
            task_track_started=True,
        )

        logger.info(f"Celery app configured with broker: {broker_url}")
        return app

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status and result of a background task.

        Args:
            task_id: The task ID returned by async task submission

        Returns:
            Dictionary with task status, result, and metadata
        """
        try:
            result = AsyncResult(task_id, app=self.app)

            status_info = {
                'task_id': task_id,
                'status': result.status,
                'current': result.current,
                'total': result.total,
                'info': result.info,
                'timestamp': datetime.now().isoformat()
            }

            # Add result if task is complete
            if result.ready():
                if result.successful():
                    status_info['result'] = result.result
                else:
                    status_info['error'] = str(result.result) if result.result else "Unknown error"
                    status_info['traceback'] = result.traceback

            return status_info

        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {str(e)}")
            return {
                'task_id': task_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_active_tasks(self) -> Dict[str, Any]:
        """
        Get information about currently active tasks across all workers.

        Returns:
            Dictionary with active task information
        """
        try:
            inspect = self.app.control.inspect()

            active_tasks = inspect.active()
            scheduled_tasks = inspect.scheduled()
            reserved_tasks = inspect.reserved()

            return {
                'active': active_tasks or {},
                'scheduled': scheduled_tasks or {},
                'reserved': reserved_tasks or {},
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting active tasks: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get statistics about Celery workers.

        Returns:
            Dictionary with worker statistics
        """
        try:
            inspect = self.app.control.inspect()

            stats = inspect.stats()
            registered = inspect.registered()
            active_queues = inspect.active_queues()

            return {
                'stats': stats or {},
                'registered': registered or {},
                'active_queues': active_queues or {},
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting worker stats: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: The task ID to cancel

        Returns:
            True if task was successfully cancelled, False otherwise
        """
        try:
            result = AsyncResult(task_id, app=self.app)
            result.revoke(terminate=True)

            logger.info(f"Cancelled task: {task_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {str(e)}")
            return False

    def get_queue_length(self, queue_name: str) -> int:
        """
        Get the length of a specific queue.

        Args:
            queue_name: Name of the queue to check

        Returns:
            Number of tasks in the queue
        """
        try:
            with self.app.connection_or_acquire() as conn:
                queue_length = conn.default_channel.queue_declare(
                    queue=queue_name, passive=True
                ).message_count

            return queue_length

        except Exception as e:
            logger.error(f"Error getting queue length for {queue_name}: {str(e)}")
            return 0

    def get_all_queue_lengths(self) -> Dict[str, int]:
        """
        Get lengths of all configured queues.

        Returns:
            Dictionary with queue names and their lengths
        """
        queues = ['ingestion', 'maintenance', 'rag_operations']
        lengths = {}

        for queue in queues:
            lengths[queue] = self.get_queue_length(queue)

        return lengths

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Celery infrastructure.

        Returns:
            Dictionary with health status information
        """
        try:
            # Test broker connection
            inspect = self.app.control.inspect()
            stats = inspect.stats()

            health = {
                'celery_available': True,
                'broker_connected': True,
                'workers_available': len(stats) > 0 if stats else False,
                'active_workers': list(stats.keys()) if stats else [],
                'queue_lengths': self.get_all_queue_lengths(),
                'timestamp': datetime.now().isoformat()
            }

            return health

        except Exception as e:
            logger.error(f"Celery health check failed: {str(e)}")
            return {
                'celery_available': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global Celery service instance
celery_service = CeleryService()

# Export Celery app for task registration
celery_app = celery_service.app