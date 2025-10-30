import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SimpleNodeParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.logger import chat_logger
from config.settings import settings
from services.chunking_service import chunking_service
from services.embedding_service import embedding_service
from services.document_tracking_service import document_tracking_service

try:
    from llama_index.readers.file import PDFReader, DocxReader
    LLAMAINDEX_FILE_READERS_AVAILABLE = True
except ImportError:
    LLAMAINDEX_FILE_READERS_AVAILABLE = False
    chat_logger.warning("LlamaIndex file readers not available, using fallback")


class IngestionService:
    """Service for document ingestion using LlamaIndex with async processing"""

    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.md']
        self.max_file_size = getattr(settings, 'MAX_FILE_SIZE_MB', 50) * 1024 * 1024  # 50MB default
        self.batch_size = getattr(settings, 'INGESTION_BATCH_SIZE', 10)

    async def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate file before ingestion
        Returns (is_valid, error_message)
        """
        try:
            path = Path(file_path)

            if not path.exists():
                return False, f"File does not exist: {file_path}"

            if not path.is_file():
                return False, f"Path is not a file: {file_path}"

            if path.suffix.lower() not in self.supported_formats:
                return False, f"Unsupported file format: {path.suffix}. Supported: {self.supported_formats}"

            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                return False, f"File too large: {file_size / (1024*1024):.1f}MB (max: {self.max_file_size / (1024*1024):.1f}MB)"

            # Check if file is already processed
            file_hash = await self._get_file_hash(file_path)
            existing_doc = await document_tracking_service.get_document_by_hash(file_hash)
            if existing_doc:
                return False, f"Document already exists: {existing_doc['filename']}"

            return True, ""

        except Exception as e:
            chat_logger.error(f"File validation error for {file_path}: {str(e)}")
            return False, f"Validation error: {str(e)}"

    async def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash for duplicate detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def load_document_async(self, file_path: str, user_token: str) -> Document:
        """
        Load document using LlamaIndex with async processing
        """
        try:
            chat_logger.info(f"Loading document: {file_path}")

            # Validate file first
            is_valid, error_msg = await self.validate_file(file_path)
            if not is_valid:
                raise ValueError(error_msg)

            path = Path(file_path)

            if LLAMAINDEX_FILE_READERS_AVAILABLE:
                # Use specialized readers for better format support
                if path.suffix.lower() == '.pdf':
                    reader = PDFReader()
                    documents = await reader.aload_data(file_path)
                elif path.suffix.lower() == '.docx':
                    reader = DocxReader()
                    documents = await reader.aload_data(file_path)
                else:
                    # For txt/md, use SimpleDirectoryReader
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    documents = await reader.aload_data()
            else:
                # Fallback to basic loading
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                documents = [Document(
                    text=content,
                    metadata={
                        'file_path': file_path,
                        'file_name': path.name,
                        'file_size': path.stat().st_size,
                        'file_type': path.suffix,
                        'user_token': user_token,
                        'ingestion_timestamp': datetime.now().isoformat()
                    }
                )]

            if not documents:
                raise ValueError(f"No content loaded from {file_path}")

            # Enhance metadata
            for doc in documents:
                doc.metadata.update({
                    'user_token': user_token,
                    'ingestion_timestamp': datetime.now().isoformat(),
                    'file_hash': await self._get_file_hash(file_path)
                })

            chat_logger.info(f"Successfully loaded document with {len(documents)} sections")
            return documents[0] if len(documents) == 1 else documents

        except Exception as e:
            chat_logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def process_document_async(self, file_path: str, user_token: str) -> Dict[str, Any]:
        """
        Process document end-to-end: load, chunk, embed, and store
        """
        try:
            chat_logger.info(f"Starting document processing for {file_path}")

            # Load document
            document = await self.load_document_async(file_path, user_token)
            if isinstance(document, list):
                # Handle multiple documents if returned
                document = document[0]

            # Chunk the document
            chunks = await chunking_service.chunk_with_rich_metadata_async(
                document.text, document.metadata.get('file_name', 'unknown')
            )

            # Generate embeddings for chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = await embedding_service.generate_embeddings_batch(chunk_texts)

            # Prepare data for storage
            processed_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                processed_chunk = {
                    'id': f"{user_token}_{document.metadata['file_hash']}_{i}",
                    'text': chunk['text'],
                    'metadata': {
                        **chunk['metadata'],
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'embedding': embedding
                    },
                    'user_token': user_token
                }
                processed_chunks.append(processed_chunk)

            # Track document
            doc_info = {
                'filename': document.metadata['file_name'],
                'file_path': file_path,
                'file_hash': document.metadata['file_hash'],
                'file_size': document.metadata.get('file_size', 0),
                'user_token': user_token,
                'chunk_count': len(processed_chunks),
                'ingestion_timestamp': document.metadata['ingestion_timestamp'],
                'status': 'processed'
            }

            await document_tracking_service.track_document(**doc_info)

            result = {
                'document_info': doc_info,
                'chunks': processed_chunks,
                'total_chunks': len(processed_chunks)
            }

            chat_logger.info(f"Document processing completed: {len(processed_chunks)} chunks created")
            return result

        except Exception as e:
            chat_logger.error(f"Error processing document {file_path}: {str(e)}")

            # Track failed document
            try:
                path = Path(file_path)
                file_hash = await self._get_file_hash(file_path)
                await document_tracking_service.track_document(
                    filename=path.name,
                    file_path=file_path,
                    file_hash=file_hash,
                    file_size=path.stat().st_size,
                    user_token=user_token,
                    chunk_count=0,
                    ingestion_timestamp=datetime.now().isoformat(),
                    status='failed',
                    error=str(e)
                )
            except Exception as track_error:
                chat_logger.error(f"Error tracking failed document: {str(track_error)}")

            raise

    async def process_documents_batch(self, file_paths: List[str], user_token: str) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch with concurrency control
        """
        if not file_paths:
            return []

        chat_logger.info(f"Processing batch of {len(file_paths)} documents")

        semaphore = asyncio.Semaphore(self.batch_size)  # Limit concurrent processing

        async def process_with_semaphore(file_path: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_document_async(file_path, user_token)

        # Process in batches to avoid overwhelming resources
        results = []
        for i in range(0, len(file_paths), self.batch_size):
            batch_files = file_paths[i:i + self.batch_size]
            chat_logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch_files)} files")

            tasks = [process_with_semaphore(file_path) for file_path in batch_files]
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        chat_logger.error(f"Failed to process {batch_files[j]}: {str(result)}")
                        results.append({
                            'error': str(result),
                            'file_path': batch_files[j],
                            'success': False
                        })
                    else:
                        results.append(result)

            except Exception as e:
                chat_logger.error(f"Batch processing failed: {str(e)}")
                # Continue with next batch

        successful = sum(1 for r in results if not r.get('error'))
        chat_logger.info(f"Batch processing completed: {successful}/{len(results)} successful")
        return results

    async def get_ingestion_status(self, user_token: str) -> Dict[str, Any]:
        """
        Get ingestion statistics for a user
        """
        try:
            documents = await document_tracking_service.get_user_documents(user_token)

            stats = {
                'total_documents': len(documents),
                'processed_documents': sum(1 for d in documents if d.get('status') == 'processed'),
                'failed_documents': sum(1 for d in documents if d.get('status') == 'failed'),
                'total_chunks': sum(d.get('chunk_count', 0) for d in documents),
                'recent_documents': documents[:5]  # Last 5 documents
            }

            return stats

        except Exception as e:
            chat_logger.error(f"Error getting ingestion status: {str(e)}")
            return {'error': str(e)}


# Global instance
ingestion_service = IngestionService()