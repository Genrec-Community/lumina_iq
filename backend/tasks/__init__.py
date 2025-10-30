"""
Celery tasks package for background operations.
Contains task definitions for document processing, caching, and RAG operations.
"""

from .celery_app import app as celery_app

__all__ = ['celery_app']