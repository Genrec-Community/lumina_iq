"""
Enhanced logging configuration with Rich formatting and better log management.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Suppress Google Cloud ALTS warnings at module level
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install
    from rich.theme import Theme

    RICH_AVAILABLE = True

    # Install rich traceback handler
    install(show_locals=False)

    # Custom theme for better readability
    custom_theme = Theme(
        {
            "info": "cyan",
            "warning": "yellow",
            "error": "bold red",
            "success": "bold green",
            "service": "bold blue",
            "timestamp": "dim white",
        }
    )

    console = Console(theme=custom_theme)

except ImportError:
    RICH_AVAILABLE = False
    console = None


def configure_logging():
    """Configure enhanced logging with Rich formatting"""

    # Only configure logging once per process
    if hasattr(logging, "_configured_for_multiworker"):
        return

    # Suppress noisy loggers
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("google.auth.transport").setLevel(logging.WARNING)
    logging.getLogger("google.auth._default").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

    # Get worker ID from environment
    worker_id = os.getenv("UVICORN_WORKER_ID", "1")

    # Get log level and format from environment
    log_level_str = os.getenv("LOG_LEVEL", "DEBUG").upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    # Map string to logging level
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(log_level_str, logging.DEBUG)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up console handler for all workers to ensure logs are visible
    if RICH_AVAILABLE:
        # Use Rich handler for better formatting - no custom formatter needed
        rich_handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            show_level=False,
            rich_tracebacks=True,
        )
        console_handler = rich_handler
    else:
        # Fallback to standard handler
        console_handler = logging.StreamHandler()

    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Removed file handler to use only RichHandler as requested

    # Configure specific loggers to prevent spam
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(log_level)  # Use the same level as root

    # Mark as configured
    logging._configured_for_multiworker = True

    return worker_id


def get_logger(name: str):
    """Get a logger with enhanced configuration"""
    configure_logging()
    return logging.getLogger(name)
