"""
NLTK Initialization Utility
Ensures NLTK data is properly downloaded and initialized for LlamaIndex and other services.
Handles missing or corrupted NLTK data gracefully.
"""

import os
import shutil
import logging
from pathlib import Path
import nltk
from utils.logger import get_logger

# Use enhanced logger
logger = get_logger("nltk_init")


def get_nltk_data_path():
    """Get the NLTK data directory path"""
    # Use environment variable if set, otherwise default to user home
    nltk_data = os.getenv("NLTK_DATA", os.path.expanduser("~/nltk_data"))
    return Path(nltk_data)


def ensure_nltk_data_exists():
    """Ensure NLTK data directory exists"""
    nltk_data_path = get_nltk_data_path()
    nltk_data_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"NLTK data directory: {nltk_data_path}")
    return nltk_data_path


def is_nltk_data_corrupted(file_path):
    """Check if NLTK data file is corrupted (e.g., not a valid zip file)"""
    try:
        import zipfile

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.testzip()
        return False
    except (zipfile.BadZipFile, zipfile.LargeZipFile, FileNotFoundError):
        return True
    except Exception as e:
        logger.warning(f"Error checking NLTK data file {file_path}: {e}")
        return True


def download_nltk_resource(resource_name, force=False):
    """Download a specific NLTK resource with error handling"""
    try:
        logger.info(f"Downloading NLTK resource: {resource_name}")
        nltk.download(resource_name, quiet=True, force=force)
        logger.info(f"Successfully downloaded NLTK resource: {resource_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK resource {resource_name}: {e}")
        return False


def cleanup_corrupted_nltk_data():
    """Remove corrupted NLTK data files"""
    nltk_data_path = get_nltk_data_path()

    if not nltk_data_path.exists():
        return

    corrupted_files = []

    # Check common NLTK data files for corruption
    for pattern in ["tokenizers/punkt*", "tokenizers/punkt_tab*", "corpora/stopwords*"]:
        for file_path in nltk_data_path.glob(pattern):
            if file_path.is_file() and is_nltk_data_corrupted(file_path):
                logger.warning(f"Found corrupted NLTK data file: {file_path}")
                corrupted_files.append(file_path)

    # Remove corrupted files
    for file_path in corrupted_files:
        try:
            file_path.unlink()
            logger.info(f"Removed corrupted NLTK data file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove corrupted file {file_path}: {e}")


def initialize_nltk_data():
    """
    Initialize NLTK data with required resources for LlamaIndex and RAG services.
    Downloads missing resources and handles corrupted files.
    """
    logger.info("Initializing NLTK data...")

    # Ensure NLTK data directory exists
    nltk_data_path = ensure_nltk_data_exists()

    # Clean up any corrupted data
    cleanup_corrupted_nltk_data()

    # Required NLTK resources for LlamaIndex and text processing
    required_resources = [
        "punkt_tab",  # Sentence tokenizer (required by LlamaIndex)
        "stopwords",  # Stop words for text processing
        "wordnet",  # WordNet for semantic analysis
        "averaged_perceptron_tagger",  # POS tagger
    ]

    missing_resources = []

    # Check which resources are missing or corrupted
    for resource in required_resources:
        try:
            # Try to find the resource
            nltk.data.find(
                f"tokenizers/{resource}"
                if resource == "punkt_tab"
                else f"corpora/{resource}"
                if resource in ["stopwords", "wordnet"]
                else f"taggers/{resource}"
            )
            logger.debug(f"NLTK resource {resource} is available")
        except LookupError:
            missing_resources.append(resource)
            logger.warning(f"NLTK resource {resource} is missing")

    # Download missing resources
    if missing_resources:
        logger.info(f"Downloading missing NLTK resources: {missing_resources}")
        for resource in missing_resources:
            success = download_nltk_resource(resource, force=True)
            if not success:
                logger.error(f"Failed to download NLTK resource: {resource}")
                # Continue with other resources even if one fails

    # Verify punkt_tab tokenizer specifically (critical for LlamaIndex)
    try:
        nltk.data.find("tokenizers/punkt_tab")
        logger.info("NLTK punkt_tab tokenizer is ready")
    except LookupError:
        logger.error(
            "Failed to initialize NLTK punkt_tab tokenizer - this may cause issues with LlamaIndex"
        )
        # Try one more time to download
        download_nltk_resource("punkt_tab", force=True)

    logger.info("NLTK data initialization completed")


def safe_nltk_operation(operation_name, operation_func, *args, **kwargs):
    """
    Safely execute an NLTK operation with error handling and recovery.

    Args:
        operation_name: Name of the operation for logging
        operation_func: The NLTK function to execute
        *args, **kwargs: Arguments for the function

    Returns:
        Result of the operation or None if failed
    """
    try:
        return operation_func(*args, **kwargs)
    except LookupError as e:
        logger.warning(f"NLTK resource missing for {operation_name}: {e}")
        # Try to download the required resource
        if "punkt" in str(e).lower():
            download_nltk_resource("punkt_tab")
        elif "stopwords" in str(e).lower():
            download_nltk_resource("stopwords")
        elif "wordnet" in str(e).lower():
            download_nltk_resource("wordnet")

        # Retry the operation
        try:
            return operation_func(*args, **kwargs)
        except Exception as e2:
            logger.error(
                f"Failed to execute NLTK operation {operation_name} after retry: {e2}"
            )
            return None
    except Exception as e:
        logger.error(f"Unexpected error in NLTK operation {operation_name}: {e}")
        return None


# Initialize NLTK data when module is imported
try:
    initialize_nltk_data()
except Exception as e:
    logger.error(f"Failed to initialize NLTK data: {e}")
