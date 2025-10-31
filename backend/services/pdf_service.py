"""
PDF service for document processing and management.
Handles PDF uploads, selection, and text extraction.
"""

from fastapi import UploadFile, HTTPException
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import pdfplumber
from PyPDF2 import PdfReader
import hashlib
from models.pdf import PDFInfo, PDFListResponse, PDFUploadResponse, PDFMetadata
from config.settings import settings
from utils.logger import get_logger
from utils.storage import pdf_contexts, pdf_metadata, storage_manager
from utils.file_hash import file_hash_service
import datetime

logger = get_logger("pdf_service")


class PDFService:
    """Service for PDF document management"""

    @staticmethod
    async def list_pdfs(
        offset: int = 0, limit: int = 20, search: Optional[str] = None
    ) -> PDFListResponse:
        """List all PDFs in the books directory with pagination and search"""
        try:
            books_dir = Path(settings.BOOKS_DIR)

            if not books_dir.exists():
                logger.warning(f"Books directory does not exist: {books_dir}")
                return PDFListResponse(items=[], total=0, offset=offset, limit=limit)

            # Get all PDF files
            pdf_files = list(books_dir.glob("*.pdf"))

            # Apply search filter if provided
            if search:
                search_lower = search.lower()
                pdf_files = [
                    f for f in pdf_files if search_lower in f.name.lower()
                ]

            total = len(pdf_files)

            # Sort by modification time (newest first)
            pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Apply pagination
            paginated_files = pdf_files[offset : offset + limit]

            # Extract metadata for each PDF
            items = []
            for pdf_path in paginated_files:
                try:
                    metadata = await PDFService._extract_basic_metadata(pdf_path)
                    items.append(metadata)
                except Exception as e:
                    logger.error(
                        f"Failed to extract metadata for {pdf_path.name}: {str(e)}"
                    )
                    # Add basic info even if metadata extraction fails
                    items.append(
                        PDFInfo(
                            filename=pdf_path.name,
                            title=pdf_path.stem,
                            author="Unknown",
                            pages=0,
                            file_size=pdf_path.stat().st_size,
                            file_path=str(pdf_path),
                        )
                    )

            return PDFListResponse(
                items=items, total=total, offset=offset, limit=limit
            )

        except Exception as e:
            logger.error(
                f"Failed to list PDFs: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {str(e)}")

    @staticmethod
    async def _extract_basic_metadata(pdf_path: Path) -> PDFInfo:
        """Extract basic metadata from a PDF file"""
        try:
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                info = reader.metadata

                title = info.get("/Title", pdf_path.stem) if info else pdf_path.stem
                author = info.get("/Author", "Unknown") if info else "Unknown"
                pages = len(reader.pages)

            return PDFInfo(
                filename=pdf_path.name,
                title=title,
                author=author,
                pages=pages,
                file_size=pdf_path.stat().st_size,
                file_path=str(pdf_path),
            )

        except Exception as e:
            logger.error(f"Metadata extraction failed for {pdf_path.name}: {str(e)}")
            raise

    @staticmethod
    async def select_pdf(filename: str, user_session: str) -> Dict[str, Any]:
        """Select a PDF for the current user session"""
        try:
            books_dir = Path(settings.BOOKS_DIR)
            file_path = books_dir / filename

            if not file_path.exists():
                raise HTTPException(status_code=404, detail="PDF file not found")

            if not file_path.suffix.lower() == ".pdf":
                raise HTTPException(status_code=400, detail="File is not a PDF")

            # Extract text from PDF
            text = await PDFService._extract_text(file_path)

            # Extract metadata
            metadata = await PDFService.get_pdf_metadata(str(file_path))

            # Store in session context
            storage_manager.safe_set(
                pdf_contexts,
                user_session,
                {
                    "filename": filename,
                    "text": text,
                    "file_path": str(file_path),
                },
            )

            storage_manager.safe_set(pdf_metadata, user_session, metadata)

            logger.info(
                f"PDF selected: {filename}",
                extra={
                    "extra_fields": {
                        "user_session": user_session,
                        "text_length": len(text),
                    }
                },
            )

            return {
                "message": f"PDF '{filename}' selected successfully",
                "filename": filename,
                "metadata": metadata,
                "text_length": len(text),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to select PDF: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to select PDF: {str(e)}"
            )

    @staticmethod
    async def upload_pdf(file: UploadFile, user_session: str) -> PDFUploadResponse:
        """Upload a new PDF file"""
        try:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(
                    status_code=400, detail="Only PDF files are allowed"
                )

            books_dir = Path(settings.BOOKS_DIR)
            books_dir.mkdir(parents=True, exist_ok=True)

            file_path = books_dir / file.filename

            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # Extract text and metadata
            text = await PDFService._extract_text(file_path)
            metadata = await PDFService.get_pdf_metadata(str(file_path))

            # Auto-select the uploaded PDF
            storage_manager.safe_set(
                pdf_contexts,
                user_session,
                {
                    "filename": file.filename,
                    "text": text,
                    "file_path": str(file_path),
                },
            )

            storage_manager.safe_set(pdf_metadata, user_session, metadata)

            logger.info(
                f"PDF uploaded: {file.filename}",
                extra={
                    "extra_fields": {
                        "user_session": user_session,
                        "file_size": len(content),
                    }
                },
            )

            return PDFUploadResponse(
                message=f"PDF '{file.filename}' uploaded and selected successfully",
                filename=file.filename,
                metadata=metadata,
                text_length=len(text),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to upload PDF: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to upload PDF: {str(e)}"
            )

    @staticmethod
    async def _extract_text(file_path: Path) -> str:
        """Extract text from PDF using pdfplumber"""
        try:
            text_parts = []

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            full_text = "\n\n".join(text_parts)

            logger.info(
                f"Extracted text from PDF: {file_path.name}",
                extra={"extra_fields": {"text_length": len(full_text)}},
            )

            return full_text

        except Exception as e:
            logger.error(
                f"Text extraction failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    @staticmethod
    async def get_pdf_metadata(
        file_path: str, extract_full_metadata: bool = False
    ) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            with open(file_path, "rb") as file:
                reader = PdfReader(file)
                info = reader.metadata

                metadata = {
                    "title": info.get("/Title", Path(file_path).stem) if info else Path(file_path).stem,
                    "author": info.get("/Author", "Unknown") if info else "Unknown",
                    "subject": info.get("/Subject", "") if info else "",
                    "creator": info.get("/Creator", "") if info else "",
                    "producer": info.get("/Producer", "") if info else "",
                    "creation_date": str(info.get("/CreationDate", "")) if info else "",
                    "modification_date": str(info.get("/ModDate", "")) if info else "",
                    "pages": len(reader.pages),
                    "file_size": Path(file_path).stat().st_size,
                }

                return metadata

        except Exception as e:
            logger.error(
                f"Metadata extraction failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {
                "title": Path(file_path).stem,
                "author": "Unknown",
                "pages": 0,
                "file_size": 0,
            }

    @staticmethod
    def get_pdf_info(user_session: str) -> Dict[str, Any]:
        """Get information about currently selected PDF"""
        pdf_context = storage_manager.safe_get(pdf_contexts, user_session)
        metadata = storage_manager.safe_get(pdf_metadata, user_session)

        if not pdf_context:
            raise HTTPException(status_code=400, detail="No PDF selected")

        return {
            "filename": pdf_context.get("filename"),
            "text_length": len(pdf_context.get("text", "")),
            "metadata": metadata,
        }
