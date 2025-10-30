"""
Shared Celery app instance for all task modules.
This ensures all tasks use the same configured Celery application.
"""

from backend.services.celery_service import celery_app

# Re-export the celery app for use in task modules
app = celery_app