"""
Document metadata extraction service.
Extracts metadata from various document types.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from PyPDF2 import PdfReader
import datetime
from utils.logger import get_logger
from utils.file_hash import file_hash_service

logger = get_logger("document_metadata_extractor")


class DocumentMetadataExtractor:
    """Service for extracting document metadata"""

    @staticmethod
    def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from a PDF file"""
        try:
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Get file stats
            stats = path.stat()

            # Extract PDF-specific metadata
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                info = reader.metadata

                metadata = {
                    "filename": path.name,
                    "file_path": str(path.absolute()),
                    "file_size": stats.st_size,
                    "file_size_mb": round(stats.st_size / (1024 * 1024), 2),
                    "created_time": datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
                    "modified_time": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "file_hash": file_hash_service.calculate_file_hash(file_path),
                    "pages": len(reader.pages),
                    "title": info.get("/Title", path.stem) if info else path.stem,
                    "author": info.get("/Author", "Unknown") if info else "Unknown",
                    "subject": info.get("/Subject", "") if info else "",
                    "creator": info.get("/Creator", "") if info else "",
                    "producer": info.get("/Producer", "") if info else "",
                    "creation_date": str(info.get("/CreationDate", "")) if info else "",
                    "modification_date": str(info.get("/ModDate", "")) if info else "",
                }

                logger.info(f"Extracted metadata from PDF: {path.name}")
                return metadata

        except Exception as e:
            logger.error(f"PDF metadata extraction failed: {str(e)}")
            return {
                "filename": Path(file_path).name,
                "error": str(e),
            }

    @staticmethod
    def extract_text_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from a text file"""
        try:
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            stats = path.stat()

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                line_count = len(content.split("\n"))
                word_count = len(content.split())
                char_count = len(content)

            metadata = {
                "filename": path.name,
                "file_path": str(path.absolute()),
                "file_size": stats.st_size,
                "created_time": datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified_time": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "file_hash": file_hash_service.calculate_file_hash(file_path),
                "line_count": line_count,
                "word_count": word_count,
                "char_count": char_count,
            }

            logger.info(f"Extracted metadata from text file: {path.name}")
            return metadata

        except Exception as e:
            logger.error(f"Text metadata extraction failed: {str(e)}")
            return {
                "filename": Path(file_path).name,
                "error": str(e),
            }

    @staticmethod
    def extract_generic_metadata(file_path: str) -> Dict[str, Any]:
        """Extract generic file metadata"""
        try:
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            stats = path.stat()

            metadata = {
                "filename": path.name,
                "file_path": str(path.absolute()),
                "file_extension": path.suffix,
                "file_size": stats.st_size,
                "file_size_mb": round(stats.st_size / (1024 * 1024), 2),
                "created_time": datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified_time": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "file_hash": file_hash_service.calculate_file_hash(file_path),
            }

            logger.info(f"Extracted generic metadata: {path.name}")
            return metadata

        except Exception as e:
            logger.error(f"Generic metadata extraction failed: {str(e)}")
            return {
                "filename": Path(file_path).name if file_path else "unknown",
                "error": str(e),
            }


# Global instance
document_metadata_extractor = DocumentMetadataExtractor()
