from typing import List, Dict, Any, Optional
from services.embedding_service import EmbeddingService
from services.qdrant_service import qdrant_service
from services.chunking_service import chunking_service
from services.document_tracking_service import document_tracking_service
from services.retrieval_strategy_manager import retrieval_strategy_manager
from utils.file_hash import file_hash_service
from utils.logger import get_logger

# Use enhanced logger
logger = get_logger("rag_service")
import asyncio


class RAGService:
    """Service for RAG (Retrieval Augmented Generation)"""

    @staticmethod
    async def index_document(
        filename: str, content: str, token: str, file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index a document for RAG retrieval with duplicate detection

        Args:
            filename: Name of the document
            content: Full text content of the document
            token: User session token (user_id)
            file_path: Optional path to file for hash calculation

        Returns:
            Dictionary with indexing results
        """
        try:
            print(
                f"[DEBUG] Starting RAG indexing for {filename} with token {token[:12]} and file_path {file_path}"
            )
            logger.info(
                f"Starting document indexing: filename={filename}, content_length={len(content)}, user_id={token[:12]}, file_path={file_path}"
            )

            # Calculate file hash for duplicate detection
            file_hash = None
            file_size = len(content.encode("utf-8"))

            if file_path:
                file_hash = file_hash_service.calculate_file_hash(file_path)
                logger.info(
                    f"Calculated file hash: filename={filename}, file_path={file_path}, hash={file_hash[:16] if file_hash else 'FAILED'}"
                )
            else:
                # Calculate hash from content if no file path
                file_hash = file_hash_service.calculate_content_hash(
                    content.encode("utf-8")
                )
                logger.info(
                    f"Calculated content hash: filename={filename}, hash={file_hash[:16] if file_hash else 'FAILED'}"
                )

            if not file_hash:
                logger.warning(
                    "Could not calculate file hash, proceeding without duplicate check",
                    filename=filename,
                )

            # Check if this exact file was already uploaded by this user
            if file_hash:
                existing_doc = document_tracking_service.check_document_exists(
                    token, file_hash
                )
                logger.info(
                    f"Document tracking check result: filename={filename}, hash={file_hash[:16]}, exists={existing_doc is not None}, existing_filename={existing_doc['filename'] if existing_doc else None}"
                )
                if existing_doc:
                    logger.info(
                        f"Duplicate document detected: filename={filename}, original_filename={existing_doc['filename']}, hash={file_hash[:16]}"
                    )
                    return {
                        "status": "duplicate",
                        "filename": filename,
                        "original_filename": existing_doc["filename"],
                        "upload_date": existing_doc["upload_date"],
                        "message": f"This document was already uploaded as '{existing_doc['filename']}' on {existing_doc['upload_date'][:10]}. Using existing index.",
                        "chunk_count": existing_doc["chunk_count"],
                    }

            # Check if already indexed in Qdrant (fallback check)
            is_indexed = await qdrant_service.check_document_indexed(filename, token)
            logger.info(
                "Qdrant indexing check result",
            )
            logger.debug(
                f"Qdrant indexing check details: filename={filename}, user_id={token[:12]}, is_indexed={is_indexed}"
            )
            if is_indexed:
                logger.info(f"Document already indexed in Qdrant: {filename}")
                # Still add to tracking if not there
                if file_hash:
                    document_tracking_service.add_document(
                        token, filename, file_hash, file_size
                    )
                return {
                    "status": "already_indexed",
                    "filename": filename,
                    "message": "Document is already indexed",
                }

            # Chunk the document WITH RICH METADATA
            chunks_with_metadata = chunking_service.chunk_with_rich_metadata(
                text=content, document_name=filename, chunk_size=1000, overlap=200
            )

            if not chunks_with_metadata:
                logger.warning("No chunks created from content", filename=filename)
                return {
                    "status": "error",
                    "filename": filename,
                    "message": "Failed to create chunks from document",
                }

            # Extract just the text for embedding
            chunks_text = [c["text"] for c in chunks_with_metadata]
            metadata_list = [c["metadata"] for c in chunks_with_metadata]

            logger.info(
                f"Created {len(chunks_text)} chunks with rich metadata: {filename}"
            )

            # Generate embeddings for all chunks
            embeddings = await EmbeddingService.generate_embeddings_batch(chunks_text)

            logger.info(f"Generated {len(embeddings)} embeddings: {filename}")

            # Index in Qdrant WITH METADATA
            num_indexed = await qdrant_service.index_document(
                filename=filename,
                chunks=chunks_text,
                embeddings=embeddings,
                token=token,
                metadata_list=metadata_list,  # Pass rich metadata
            )

            # Add to document tracking database
            if file_hash:
                document_tracking_service.add_document(
                    user_id=token,
                    filename=filename,
                    file_hash=file_hash,
                    file_size=file_size,
                    chunk_count=num_indexed,
                )
                logger.info(
                    f"Document added to tracking database: filename={filename}, hash={file_hash[:16]}"
                )

            # Extract metadata statistics
            chapters = set(
                m.get("chapter_number")
                for m in metadata_list
                if m.get("chapter_number")
            )
            sections = set(
                m.get("section_number")
                for m in metadata_list
                if m.get("section_number")
            )

            return {
                "status": "success",
                "filename": filename,
                "num_chunks": len(chunks_text),
                "num_indexed": num_indexed,
                "file_hash": file_hash[:16] if file_hash else None,
                "chapters_found": len(chapters),
                "sections_found": len(sections),
                "message": f"Successfully indexed {num_indexed} chunks with metadata ({len(chapters)} chapters, {len(sections)} sections)",
            }

        except Exception as e:
            logger.error(
                f"Failed to index document: filename={filename}, error={str(e)}"
            )
            return {
                "status": "error",
                "filename": filename,
                "message": f"Failed to index document: {str(e)}",
            }

    @staticmethod
    async def retrieve_context(
        query: str,
        token: str,
        filename: str = None,
        top_k: int = 5,
        use_case: str = None,  # New parameter for explicit use case
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context chunks for a query using ADVANCED RAG with strategy selection

        Args:
            query: User's query
            token: User session token
            filename: Optional filename to filter by
            top_k: Number of top results to retrieve
            use_case: Optional explicit use case (chat, evaluation, qa_generation, notes)

        Returns:
            Dictionary with retrieved chunks and metadata
        """
        try:
            logger.info(
                f"Retrieving context with ADVANCED RAG: query_length={len(query)}, top_k={top_k}, filename={filename}, use_case={use_case}"
            )

            # Use the advanced retrieval strategy manager
            result = await retrieval_strategy_manager.retrieve(
                query=query,
                token=token,
                filename=filename,
                use_case=use_case,
                top_k=top_k,
            )

            logger.info(
                f"Advanced retrieval completed: strategy={result.get('strategy')}, num_chunks={result.get('num_chunks', 0)}"
            )

            return result

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to retrieve context: query={query[:100]}, error={str(e)}, error_type={type(e).__name__}, traceback={traceback.format_exc()}"
            )
            return {
                "status": "error",
                "chunks": [],
                "context": "",
                "num_chunks": 0,
                "message": f"Failed to retrieve context: {str(e)}",
            }

    @staticmethod
    async def delete_document_index(filename: str, token: str):
        """Delete document index from Qdrant"""
        try:
            await qdrant_service.delete_document_chunks(filename, token)
            logger.info(f"Deleted document index: {filename}")
            return {"status": "success", "message": f"Deleted index for {filename}"}
        except Exception as e:
            logger.error(
                f"Failed to delete document index: filename={filename}, error={str(e)}"
            )
            return {"status": "error", "message": f"Failed to delete index: {str(e)}"}

    @staticmethod
    async def check_indexing_status(filename: str, token: str) -> bool:
        """Check if a document is indexed"""
        try:
            return await qdrant_service.check_document_indexed(filename, token)
        except Exception as e:
            logger.error(
                f"Failed to check indexing status: filename={filename}, error={str(e)}"
            )
            return False


rag_service = RAGService()
