"""
Utility to suppress noisy warnings and logs from third-party libraries
"""

import os
import sys
import warnings
import logging


def suppress_third_party_warnings():
    """Suppress noisy warnings from third-party libraries"""

    # Suppress Google Cloud ALTS warnings
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

    # Suppress other Google Auth warnings
    os.environ["GOOGLE_CLOUD_PROJECT"] = ""

    # Suppress specific warning categories
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")
    warnings.filterwarnings("ignore", category=UserWarning, module="google")
    warnings.filterwarnings("ignore", category=FutureWarning, module="google")

    # Suppress PyPDF2 warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="PyPDF2")
    warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2")

    # Suppress other common warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pdfplumber")
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

    # Suppress NLTK warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="nltk")

    # Configure specific loggers to be less verbose
    noisy_loggers = [
        "google.auth",
        "google.auth.transport",
        "google.auth._default",
        "google.cloud",
        "grpc",
        "urllib3",
        "requests",
        "httpx",
        "httpcore",
        "uvicorn.access",
    ]

    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False


# Call this function at module import
suppress_third_party_warnings()
