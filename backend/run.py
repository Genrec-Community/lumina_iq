#!/usr/bin/env python3
"""
Run script for the Learning App FastAPI backend.
This script starts the FastAPI server with uvicorn.
"""

import uvicorn
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from utils.ip_detector import setup_frontend_env
from utils.logger import app_logger
from utils.logger import mutate_logger

logger = app_logger.getChild("api")


def setup_other_loggers():
    """Set up other module-specific loggers if needed."""
    mutate_logger("uvicorn", logging_level=settings.LOG_LEVEL)
    mutate_logger("uvicorn.error", logging_level=settings.LOG_LEVEL)
    mutate_logger("uvicorn.access", logging_level=settings.LOG_LEVEL)
    mutate_logger("urllib3", logging_level=settings.LOG_LEVEL)
    mutate_logger("qdrant_client", logging_level=settings.LOG_LEVEL)
    mutate_logger("llama_index", logging_level=settings.LOG_LEVEL)
    mutate_logger("httpx", logging_level=settings.LOG_LEVEL)
    mutate_logger("asyncio", logging_level=settings.LOG_LEVEL)
    mutate_logger("sqlalchemy", logging_level=settings.LOG_LEVEL)
    mutate_logger("multipart", logging_level=settings.LOG_LEVEL)
    mutate_logger("fastapi", logging_level=settings.LOG_LEVEL)
    mutate_logger("starlette", logging_level=settings.LOG_LEVEL)
    mutate_logger("requests", logging_level=settings.LOG_LEVEL)


def main():
    """Start the FastAPI server."""
    logger.info("🚀 Starting Learning App Backend...")

    # Auto-detect IP and update frontend .env file
    logger.info("🔧 Setting up frontend environment...")
    detected_ip = setup_frontend_env(settings.PORT)

    logger.debug(f"📍 Server will run on: http://{settings.HOST}:{settings.PORT}")
    logger.debug(f"🌐 Accessible at: http://{detected_ip}:{settings.PORT}")
    logger.debug(f"📚 Books directory: {settings.BOOKS_DIR}")
    logger.debug(f"🔑 Using Together.ai model: {settings.TOGETHER_MODEL}")

    try:
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=True,
            log_level="debug",
            workers=1,  # Single worker for development with reload
            # Basic optimizations for development
            backlog=2048,
            timeout_keep_alive=5,
            limit_concurrency=500,
        )
    except KeyboardInterrupt:
        logger.warning("\n👋 Server stopped by user")
    except Exception as e:
        logger.critical(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
