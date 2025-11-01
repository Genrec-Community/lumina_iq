"""
PDF Service for Lumina IQ RAG Backend.

Handles PDF operations including listing, selection, upload, and metadata extraction.
This service bridges the routes layer with the new RAG architecture.
"""

import os
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException, UploadFile
from typing import List, Dict, Any, Optional
import aiofiles
import hashlib

from config.settings import settings
from utils.storage import pdf_contexts, pdf_metadata, storage_manager
from utils.logger import get_logger
from models.pdf import PDFInfo, PDFListResponse, PDFUploadResponse
from .document_service import document_service
from .rag_orchestrator import rag_orchestrator
from .cache_service import cache_service

logger = get_logger("pdf_service")


class PDFService:
    """Service for PDF operations and RAG integration."""

    @staticmethod
    async def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF using document service."""
        logger.info(f"Starting PDF text extraction", extra={"extra_fields": {"file_path": file_path}})

        # Check cache first
        cached_text = await cache_service.get_cached_api_response("pdf_text", {"file_path": file_path})
        if cached_text:
            logger.info("Using cached text", extra={"extra_fields": {"file_path": file_path}})
            return cached_text

        try:
            # Extract using document service
            documents = await document_service.extract_from_pdf(Path(file_path))

            # Combine all pages into single text
            text = "\n\n".join([doc.text for doc in documents])

            # Cache the extracted text
            if settings.CACHE_QUERY_RESULTS:
                await cache_service.cache_api_response(
                    "pdf_text",
                    {"file_path": file_path},
                    text
                )

            logger.info(
                "Text extraction completed",
                extra={"extra_fields": {"file_path": file_path, "text_length": len(text)}}
            )

            return text

        except Exception as e:
            logger.error(
                f"Failed to extract text from PDF: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__, "file_path": file_path}}
            )
            raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {str(e)}")

    @staticmethod
    async def get_pdf_metadata(file_path: str, extract_full_metadata: bool = False) -> Dict[str, Any]:
        """Get PDF metadata."""
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                return {"error": "File not found"}

            # Basic metadata
            metadata = {
                "title": file_path_obj.stem,
                "author": "Unknown",
                "subject": "Unknown",
                "creator": "Unknown",
                "producer": "Unknown",
                "creation_date": "Unknown",
                "modification_date": "Unknown",
                "pages": 0,
                "file_size": file_path_obj.stat().st_size,
            }

            if extract_full_metadata:
                # Try to extract more detailed metadata using PyPDF2
                try:
                    import PyPDF2
                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        if pdf_reader.metadata:
                            metadata.update({
                                "title": pdf_reader.metadata.get("/Title", metadata["title"]),
                                "author": pdf_reader.metadata.get("/Author", "Unknown"),
                                "subject": pdf_reader.metadata.get("/Subject", "Unknown"),
                                "creator": pdf_reader.metadata.get("/Creator", "Unknown"),
                                "producer": pdf_reader.metadata.get("/Producer", "Unknown"),
                                "creation_date": str(pdf_reader.metadata.get("/CreationDate", "Unknown")),
                                "modification_date": str(pdf_reader.metadata.get("/ModDate", "Unknown")),
                            })
                        metadata["pages"] = len(pdf_reader.pages)
                except Exception as e:
                    logger.warning(f"Failed to extract full metadata: {str(e)}")

            return metadata

        except Exception as e:
            logger.error(f"Failed to get PDF metadata: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    async def list_pdfs(offset: int = 0, limit: int = 20, search: Optional[str] = None) -> PDFListResponse:
        """List PDFs in the books folder with pagination and optional search."""
        try:
            books_dir = Path(settings.BOOKS_DIR)
            if not books_dir.exists():
                books_dir.mkdir(parents=True, exist_ok=True)
                return PDFListResponse(items=[], total=0, offset=offset, limit=limit)

            # Get all PDFs
            all_pdfs = []
            for file_path in books_dir.glob("*.pdf"):
                try:
                    metadata = await PDFService.get_pdf_metadata(str(file_path), extract_full_metadata=False)
                    pdf_info = PDFInfo(
                        filename=file_path.name,
                        title=metadata.get("title", "Unknown"),
                        author=metadata.get("author", "Unknown"),
                        pages=metadata.get("pages", 0),
                        file_size=metadata.get("file_size", 0),
                        file_path=str(file_path),
                    )
                    all_pdfs.append(pdf_info)
                except Exception as e:
                    logger.warning(f"Failed to process PDF {file_path.name}: {str(e)}")
                    continue

            # Apply search filter
            if search:
                search_lower = search.lower()
                filtered_pdfs = [
                    pdf for pdf in all_pdfs
                    if search_lower in pdf.title.lower() or search_lower in pdf.filename.lower()
                ]
            else:
                filtered_pdfs = all_pdfs

            # Apply pagination
            total = len(filtered_pdfs)
            paginated_pdfs = filtered_pdfs[offset:offset + limit]

            return PDFListResponse(items=paginated_pdfs, total=total, offset=offset, limit=limit)

        except Exception as e:
            logger.error(f"Failed to list PDFs: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {str(e)}")

    @staticmethod
    async def select_pdf(filename: str, token: str) -> Dict[str, Any]:
        """Select a PDF from the books folder for the session."""
        logger.info(f"Selecting PDF", extra={"extra_fields": {"filename": filename, "token": token[:12]}})

        try:
            books_dir = Path(settings.BOOKS_DIR)
            file_path = books_dir / filename

            if not file_path.exists():
                raise HTTPException(status_code=404, detail="PDF file not found")

            if not file_path.suffix.lower() == ".pdf":
                raise HTTPException(status_code=400, detail="File is not a PDF")

            # Extract text and metadata
            text_content = await PDFService.extract_text_from_pdf(str(file_path))
            metadata = await PDFService.get_pdf_metadata(str(file_path), extract_full_metadata=True)

            # Store in session
            storage_manager.safe_set(
                pdf_contexts,
                token,
                {
                    "filename": filename,
                    "content": text_content,
                    "selected_at": datetime.now().isoformat(),
                }
            )
            storage_manager.safe_set(pdf_metadata, token, metadata)

            # Index document for RAG
            logger.info(f"Indexing document with RAG orchestrator", extra={"extra_fields": {"filename": filename}})
            indexing_result = await rag_orchestrator.ingest_document(
                file_path=file_path,
                metadata={"token": token, "source": "select"}
            )

            return {
                "message": "PDF selected successfully",
                "filename": filename,
                "metadata": metadata,
                "text_length": len(text_content),
                "rag_indexing": "success" if indexing_result.get("success") else "failed",
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to select PDF: {str(e)}", extra={"extra_fields": {"error_type": type(e).__name__}})
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    @staticmethod
    def _generate_unique_filename(books_dir: Path, original_filename: str) -> str:
        """Generate a unique filename to avoid conflicts."""
        if not (books_dir / original_filename).exists():
            return original_filename

        name_part = original_filename.rsplit(".", 1)[0]
        extension = "." + original_filename.rsplit(".", 1)[1] if "." in original_filename else ""

        counter = 1
        while True:
            new_filename = f"{name_part}({counter}){extension}"
            if not (books_dir / new_filename).exists():
                return new_filename
            counter += 1

    @staticmethod
    async def upload_pdf(file: UploadFile, token: str) -> PDFUploadResponse:
        """Upload a new PDF to the books folder."""
        logger.info(f"Uploading PDF", extra={"extra_fields": {"filename": file.filename, "token": token[:12]}})

        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        try:
            # Create books directory if needed
            books_dir = Path(settings.BOOKS_DIR)
            books_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            unique_filename = PDFService._generate_unique_filename(books_dir, file.filename)
            file_path = books_dir / unique_filename

            # Save file
            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            # Extract text and metadata
            text_content = await PDFService.extract_text_from_pdf(str(file_path))
            metadata = await PDFService.get_pdf_metadata(str(file_path), extract_full_metadata=True)

            # Store in session
            storage_manager.safe_set(
                pdf_contexts,
                token,
                {
                    "filename": unique_filename,
                    "content": text_content,
                    "uploaded_at": datetime.now().isoformat(),
                }
            )
            storage_manager.safe_set(pdf_metadata, token, metadata)

            # Index document for RAG
            logger.info(f"Indexing uploaded document", extra={"extra_fields": {"filename": unique_filename}})
            indexing_result = await rag_orchestrator.ingest_document(
                file_path=file_path,
                metadata={"token": token, "source": "upload"}
            )

            return PDFUploadResponse(
                message="PDF uploaded and processed successfully",
                filename=unique_filename,
                metadata=metadata,
                text_length=len(text_content),
            )

        except HTTPException:
            raise
        except Exception as e:
            # Clean up file if processing failed
            if file_path.exists():
                file_path.unlink()
            logger.error(f"Failed to upload PDF: {str(e)}", extra={"extra_fields": {"error_type": type(e).__name__}})
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    @staticmethod
    def get_pdf_info(token: str) -> Dict[str, Any]:
        """Get info about the currently selected PDF."""
        if token not in pdf_contexts:
            raise HTTPException(status_code=400, detail="No PDF selected")

        pdf_context = pdf_contexts[token]
        metadata = pdf_metadata.get(token, {})

        return {
            "filename": pdf_context["filename"],
            "selected_at": pdf_context.get("selected_at") or pdf_context.get("uploaded_at"),
            "text_length": len(pdf_context["content"]),
            "metadata": metadata,
        }
