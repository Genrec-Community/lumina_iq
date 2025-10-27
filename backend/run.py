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


def main():
    """Start the FastAPI server."""
    print("🚀 Starting Learning App Backend...")

    # Auto-detect IP and update frontend .env file
    print("\n🔧 Setting up frontend environment...")
    detected_ip = setup_frontend_env(settings.PORT)

    print(f"\n📍 Server will run on: http://{settings.HOST}:{settings.PORT}")
    print(f"🌐 Accessible at: http://{detected_ip}:{settings.PORT}")
    print(f"📚 Books directory: {settings.BOOKS_DIR}")
    print(f"🔑 Using Together.ai model: {settings.TOGETHER_MODEL}")
    print("=" * 50)

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
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
