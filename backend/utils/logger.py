import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from config.settings import settings


# COnsoles for RichHandlers
console = Console()
file_console = Console(
    file=open(f"{datetime.today().strftime('%d_%b_%Y')}.log", "a+"), width=console.width
)

# Map string to logging level
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
log_level = log_level_map.get(settings.LOG_LEVEL.upper(), logging.WARNING)

# Global logger instances
pdf_logger = logging.getLogger("pdf_service")
cache_logger = logging.getLogger("cache")
chat_logger = logging.getLogger("chat_service")
ip_logger = logging.getLogger("ip_detector")

# Attach Handlers - Console
for each in (pdf_logger, cache_logger, chat_logger, ip_logger):
    each.root.handlers = []
    each.addHandler(RichHandler(console=console))
    each.addHandler(RichHandler(console=file_console))
    each.setLevel(log_level)
    for handler in each.handlers:
        handler.setLevel(log_level)

for each in (pdf_logger, cache_logger, chat_logger, ip_logger):
    each.info("Test info")
    each.debug("Test debug")
    each.warning("Test warning")
    each.critical("Test critical")
    each.error("Test error")


# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)
