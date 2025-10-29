import os
import pdfplumber
import aiofiles
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException, UploadFile
from typing import List
from config.settings import settings
from utils.storage import pdf_contexts, pdf_metadata, storage_manager, storage_manager
from utils.cache import cache_service
from utils.logger import get_logger

# Use enhanced logger
logger = get_logger("pdf_service")
from models.pdf import PDFInfo, PDFListResponse, PDFUploadResponse, PDFMetadata
from services.rag_service import rag_service
from services.llamaindex_service import llamaindex_service
import warnings
import asyncio

# Suppress PyPDF2 warnings when it's imported
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PyPDF2")
warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2")


class PDFService:
    @staticmethod
    def should_use_llamaindex(file_path: str) -> bool:
        """Determine if LlamaIndex should be used for this PDF based on size"""
        try:
            file_size = os.path.getsize(file_path)
            threshold = settings.LLAMAINDEX_LARGE_PDF_THRESHOLD_MB * 1024 * 1024
            return settings.LLAMAINDEX_USE_FOR_LARGE_PDFS and file_size > threshold
        except Exception:
            return False

    @staticmethod
    async def extract_text_from_pdf(file_path: str) -> str:
        logger.info("Starting PDF text extraction", file_path=file_path)

        # Check cache first
        cached_text = await cache_service.get_cached_text(file_path)
        if cached_text is not None:
            logger.info(
                "Using cached text", file_path=file_path, text_length=len(cached_text)
            )
            return cached_text

        # Cache miss - extract text from PDF
        text = ""
        logger.info("Cache miss - extracting text from PDF", file_path=file_path)

        try:
            # Lazy import PyPDF2 only when needed for text extraction
            import PyPDF2

            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                logger.info(
                    "PDF loaded successfully", file_path=file_path, pages=page_count
                )

                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        logger.debug(
                            "Page extracted", page=i + 1, text_length=len(page_text)
                        )
                    else:
                        logger.debug("Page has no text", page=i + 1)

        except Exception as e:
            logger.warning(
                "PyPDF2 extraction failed, trying pdfplumber",
                file_path=file_path,
                error=str(e),
            )
            # Fallback to pdfplumber
            try:
                with pdfplumber.open(file_path) as pdf:
                    page_count = len(pdf.pages)
                    logger.info(
                        "PDF loaded with pdfplumber",
                        file_path=file_path,
                        pages=page_count,
                    )

                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            logger.debug(
                                "Page extracted with pdfplumber",
                                page=i + 1,
                                text_length=len(page_text),
                            )
                        else:
                            logger.debug("Page has no text with pdfplumber", page=i + 1)

            except Exception as e2:
                logger.error(
                    "Both extraction methods failed",
                    file_path=file_path,
                    pypdf_error=str(e),
                    pdfplumber_error=str(e2),
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to extract text from PDF: {str(e2)}",
                )

        extracted_text = text.strip()
        logger.info(
            "Text extraction completed",
            file_path=file_path,
            extracted_length=len(extracted_text),
        )

        if len(extracted_text) < 50:
            logger.warning(
                "Very little text extracted",
                file_path=file_path,
                extracted_length=len(extracted_text),
                preview=extracted_text[:100],
            )

        # Save to cache for future use
        cache_saved = await cache_service.save_to_cache(file_path, extracted_text)
        if cache_saved:
            logger.info("Successfully cached extracted text", file_path=file_path)
        else:
            logger.warning("Failed to cache extracted text", file_path=file_path)

        return extracted_text

    @staticmethod
    async def get_pdf_metadata(
        file_path: str, extract_full_metadata: bool = False
    ) -> dict:
        """Get PDF metadata. If extract_full_metadata is False, only get basic file info without using PyPDF2"""
        basic_metadata = {
            "title": Path(file_path).stem,  # Use filename as title
            "author": "Unknown",
            "subject": "Unknown",
            "creator": "Unknown",
            "producer": "Unknown",
            "creation_date": "Unknown",
            "modification_date": "Unknown",
            "pages": 0,  # Will be set to unknown for basic metadata
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        }

        if not extract_full_metadata:
            return basic_metadata

        # Only use PyPDF2 when full metadata is explicitly requested
        try:
            import PyPDF2

            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = {
                    "title": pdf_reader.metadata.get("/Title", basic_metadata["title"])
                    if pdf_reader.metadata
                    else basic_metadata["title"],
                    "author": pdf_reader.metadata.get("/Author", "Unknown")
                    if pdf_reader.metadata
                    else "Unknown",
                    "subject": pdf_reader.metadata.get("/Subject", "Unknown")
                    if pdf_reader.metadata
                    else "Unknown",
                    "creator": pdf_reader.metadata.get("/Creator", "Unknown")
                    if pdf_reader.metadata
                    else "Unknown",
                    "producer": pdf_reader.metadata.get("/Producer", "Unknown")
                    if pdf_reader.metadata
                    else "Unknown",
                    "creation_date": str(
                        pdf_reader.metadata.get("/CreationDate", "Unknown")
                    )
                    if pdf_reader.metadata
                    else "Unknown",
                    "modification_date": str(
                        pdf_reader.metadata.get("/ModDate", "Unknown")
                    )
                    if pdf_reader.metadata
                    else "Unknown",
                    "pages": len(pdf_reader.pages),
                    "file_size": os.path.getsize(file_path),
                }
                return metadata
        except Exception as e:
            logger.warning(
                "Failed to extract full metadata", file_path=file_path, error=str(e)
            )
            return basic_metadata

    @staticmethod
    async def list_pdfs(
        offset: int = 0, limit: int = 20, search: str = None
    ) -> PDFListResponse:
        """List PDFs in the books folder with pagination and optional search"""
        books_dir = Path(settings.BOOKS_DIR)
        if not books_dir.exists():
            books_dir.mkdir(exist_ok=True)
            return PDFListResponse(items=[], total=0, offset=offset, limit=limit)

        # Get all PDFs first - use basic metadata only (no PyPDF2)
        all_pdfs = []
        for file_path in books_dir.glob("*.pdf"):
            try:
                # Only get basic metadata without using PyPDF2 for listing
                metadata = await PDFService.get_pdf_metadata(
                    str(file_path), extract_full_metadata=False
                )
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
                # Skip files that can't be processed
                continue

        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            filtered_pdfs = [
                pdf
                for pdf in all_pdfs
                if search_lower in pdf.title.lower()
                or search_lower in pdf.filename.lower()
            ]
        else:
            filtered_pdfs = all_pdfs

        # Apply pagination
        total = len(filtered_pdfs)
        start_idx = offset
        end_idx = offset + limit
        paginated_pdfs = filtered_pdfs[start_idx:end_idx]

        return PDFListResponse(
            items=paginated_pdfs, total=total, offset=offset, limit=limit
        )

    @staticmethod
    async def select_pdf(filename: str, token: str) -> dict:
        """Select a PDF from the books folder for the session"""
        print(f"[DEBUG] Starting select_pdf for {filename} with token {token[:12]}")
        books_dir = Path(settings.BOOKS_DIR)
        file_path = books_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")

        if not file_path.suffix.lower() == ".pdf":
            raise HTTPException(status_code=400, detail="File is not a PDF")

        try:
            print(f"[DEBUG] File path exists: {file_path}")
            # Extract text and metadata
            text_content = await PDFService.extract_text_from_pdf(str(file_path))
            metadata = await PDFService.get_pdf_metadata(
                str(file_path), extract_full_metadata=True
            )

            # Store in session thread-safely
            storage_manager.safe_set(
                pdf_contexts,
                token,
                {
                    "filename": filename,
                    "content": text_content,
                    "selected_at": datetime.now().isoformat(),
                },
            )
            storage_manager.safe_set(pdf_metadata, token, metadata)

            # Index document for RAG with duplicate detection
            logger.info("Starting RAG indexing for selected PDF", filename=filename)

            # Check if should use LlamaIndex for large PDFs
            use_llamaindex = PDFService.should_use_llamaindex(str(file_path))

            if use_llamaindex:
                logger.info(
                    "Using LlamaIndex for large PDF indexing", filename=filename
                )
                indexing_result = (
                    await llamaindex_service.index_document_with_llamaindex(
                        file_path=file_path, filename=filename, token=token
                    )
                )
            else:
                logger.info("Using standard RAG indexing", filename=filename)
                indexing_result = await rag_service.index_document(
                    filename=filename,
                    content=text_content,
                    token=token,
                    file_path=str(file_path),
                )
            logger.info(
                "RAG indexing completed",
                filename=filename,
                result=indexing_result.get("status"),
                is_duplicate=indexing_result.get("status") == "duplicate",
            )

            return {
                "message": "PDF selected successfully",
                "filename": filename,
                "metadata": metadata,
                "text_length": len(text_content),
                "rag_indexing": indexing_result.get("status", "unknown"),
            }

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to process PDF: {str(e)}"
            )

    @staticmethod
    def _generate_unique_filename(books_dir: Path, original_filename: str) -> str:
        """Generate a unique filename to avoid conflicts using parentheses with incremental numbers"""
        if not (books_dir / original_filename).exists():
            return original_filename

        # Split filename and extension
        name_part = original_filename.rsplit(".", 1)[0]
        extension = (
            "." + original_filename.rsplit(".", 1)[1]
            if "." in original_filename
            else ""
        )

        counter = 1
        while True:
            new_filename = f"{name_part}({counter}){extension}"
            if not (books_dir / new_filename).exists():
                return new_filename
            counter += 1

    @staticmethod
    async def upload_pdf(file: UploadFile, token: str) -> PDFUploadResponse:
        """Upload a new PDF to the books folder with automatic duplicate filename handling"""
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Create books directory if it doesn't exist
        books_dir = Path(settings.BOOKS_DIR)
        books_dir.mkdir(exist_ok=True)

        # Generate unique filename to avoid conflicts
        unique_filename = PDFService._generate_unique_filename(books_dir, file.filename)
        file_path = books_dir / unique_filename

        # Save file with unique filename
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # Extract text and metadata
        try:
            text_content = await PDFService.extract_text_from_pdf(str(file_path))
            metadata = await PDFService.get_pdf_metadata(str(file_path))

            # Store in memory thread-safely (replace with database in production)
            storage_manager.safe_set(
                pdf_contexts,
                token,
                {
                    "filename": unique_filename,
                    "content": text_content,
                    "uploaded_at": datetime.now().isoformat(),
                },
            )
            storage_manager.safe_set(pdf_metadata, token, metadata)

            # Index document for RAG with duplicate detection
            logger.info(
                "Starting RAG indexing for uploaded PDF", filename=unique_filename
            )

            # Check if should use LlamaIndex for large PDFs
            use_llamaindex = PDFService.should_use_llamaindex(str(file_path))

            if use_llamaindex:
                logger.info(
                    "Using LlamaIndex for large PDF indexing", filename=unique_filename
                )
                indexing_result = (
                    await llamaindex_service.index_document_with_llamaindex(
                        file_path=file_path,
                        filename=unique_filename,
                        token=token,
                    )
                )
            else:
                logger.info("Using standard RAG indexing", filename=unique_filename)
                indexing_result = await rag_service.index_document(
                    filename=unique_filename,
                    content=text_content,
                    token=token,
                    file_path=str(file_path),
                )
            logger.info(
                "RAG indexing completed",
                filename=unique_filename,
                result=indexing_result.get("status"),
                is_duplicate=indexing_result.get("status") == "duplicate",
            )

            # Add indexing info to response
            response_data = PDFUploadResponse(
                message="PDF uploaded and processed successfully"
                if indexing_result.get("status") != "duplicate"
                else f"PDF uploaded successfully. {indexing_result.get('message')}",
                filename=unique_filename,
                metadata=metadata,
                text_length=len(text_content),
            )

            return response_data

        except Exception as e:
            # Clean up file if processing failed
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=500, detail=f"Failed to process PDF: {str(e)}"
            )

    @staticmethod
    def get_pdf_info(token: str) -> dict:
        """Get info about the currently selected PDF"""
        if token not in pdf_contexts:
            raise HTTPException(status_code=400, detail="No PDF selected")

        pdf_context = pdf_contexts[token]
        metadata = pdf_metadata.get(token, {})

        return {
            "filename": pdf_context["filename"],
            "selected_at": pdf_context.get("selected_at")
            or pdf_context.get("uploaded_at"),
            "text_length": len(pdf_context["content"]),
            "metadata": metadata,
        }
