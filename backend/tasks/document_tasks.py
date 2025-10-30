"""
Celery tasks for background document processing operations.
Handles large file ingestion, batch processing, and document management.
"""

import asyncio
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from .celery_app import app
from backend.services.ingestion_service import ingestion_service
from backend.services.chunking_service import chunking_service
from backend.services.embedding_service import embedding_service
from backend.services.document_tracking_service import document_tracking_service
from backend.services.cache_service import cache_service
from backend.services.qdrant_service import qdrant_service
from backend.services.rag_service import rag_service
from utils.logger import get_logger

logger = get_logger("document_tasks")


@app.task(bind=True, name='tasks.document_tasks.process_single_document')
def process_single_document(self, file_path: str, user_token: str) -> Dict[str, Any]:
    """
    Process a single document asynchronously.
    Handles loading, chunking, embedding, and indexing.

    Args:
        file_path: Path to the document file
        user_token: User session token

    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"Starting background processing for document: {file_path}")

        # Update task state for progress tracking
        self.update_state(state='PROGRESS', meta={'step': 'validation'})

        # Validate file exists and is accessible
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        # Process document using ingestion service
        self.update_state(state='PROGRESS', meta={'step': 'processing'})
        result = asyncio.run(ingestion_service.process_document_async(file_path, user_token))

        # Index document in vector store
        self.update_state(state='PROGRESS', meta={'step': 'indexing'})

        # Extract content for RAG indexing
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        index_result = asyncio.run(rag_service.index_document(
            filename=os.path.basename(file_path),
            content=content,
            token=user_token,
            file_path=file_path
        ))

        # Invalidate relevant caches
        asyncio.run(cache_service.invalidate_document_data(user_token, os.path.basename(file_path)))

        logger.info(f"Successfully processed document: {file_path}")

        return {
            'status': 'success',
            'file_path': file_path,
            'processing_result': result,
            'indexing_result': index_result,
            'processed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to process document {file_path}: {str(e)}")
        # Track failed document
        try:
            asyncio.run(document_tracking_service.track_document(
                filename=os.path.basename(file_path),
                file_path=file_path,
                file_hash='',  # Will be calculated by tracking service
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                user_token=user_token,
                chunk_count=0,
                ingestion_timestamp=datetime.now().isoformat(),
                status='failed',
                error=str(e)
            ))
        except Exception as track_error:
            logger.error(f"Failed to track document error: {str(track_error)}")

        raise  # Re-raise for Celery error handling


@app.task(bind=True, name='tasks.document_tasks.process_document_batch')
def process_document_batch(self, file_paths: List[str], user_token: str) -> Dict[str, Any]:
    """
    Process multiple documents in batch.
    Provides progress tracking and error aggregation.

    Args:
        file_paths: List of file paths to process
        user_token: User session token

    Returns:
        Dictionary with batch processing results
    """
    try:
        logger.info(f"Starting batch processing for {len(file_paths)} documents")

        total_files = len(file_paths)
        processed_results = []
        successful_count = 0

        for i, file_path in enumerate(file_paths):
            try:
                # Update progress
                progress = int((i / total_files) * 100)
                self.update_state(state='PROGRESS', meta={
                    'current': i + 1,
                    'total': total_files,
                    'progress': progress,
                    'current_file': os.path.basename(file_path)
                })

                # Process individual document
                result = process_single_document.apply(args=[file_path, user_token]).get()

                if result['status'] == 'success':
                    successful_count += 1

                processed_results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {file_path} in batch: {str(e)}")
                processed_results.append({
                    'status': 'error',
                    'file_path': file_path,
                    'error': str(e)
                })

        # Final summary
        summary = {
            'status': 'completed',
            'total_files': total_files,
            'successful': successful_count,
            'failed': total_files - successful_count,
            'results': processed_results,
            'completed_at': datetime.now().isoformat()
        }

        logger.info(f"Batch processing completed: {successful_count}/{total_files} successful")
        return summary

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.document_tasks.reindex_documents')
def reindex_documents(self, user_token: str, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Re-index existing documents for a user.
    Useful after model updates or data migration.

    Args:
        user_token: User session token
        document_ids: Optional list of specific document IDs to reindex

    Returns:
        Dictionary with re-indexing results
    """
    try:
        logger.info(f"Starting document re-indexing for user: {user_token}")

        # Get user's documents
        if document_ids:
            documents = []
            for doc_id in document_ids:
                doc = asyncio.run(document_tracking_service.get_document_by_id(doc_id))
                if doc:
                    documents.append(doc)
        else:
            documents = asyncio.run(document_tracking_service.get_user_documents(user_token))

        logger.info(f"Found {len(documents)} documents to reindex")

        total_docs = len(documents)
        reindexed_count = 0
        results = []

        for i, doc in enumerate(documents):
            try:
                # Update progress
                progress = int((i / total_docs) * 100)
                self.update_state(state='PROGRESS', meta={
                    'current': i + 1,
                    'total': total_docs,
                    'progress': progress,
                    'current_document': doc.get('filename', 'unknown')
                })

                # Check if file still exists
                file_path = doc.get('file_path')
                if not file_path or not os.path.exists(file_path):
                    logger.warning(f"Document file not found, skipping: {file_path}")
                    continue

                # Delete existing index
                asyncio.run(qdrant_service.delete_document_chunks(doc['filename'], user_token))

                # Re-process document
                result = process_single_document.apply(args=[file_path, user_token]).get()

                if result['status'] == 'success':
                    reindexed_count += 1

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to reindex document {doc.get('filename')}: {str(e)}")
                results.append({
                    'status': 'error',
                    'document_id': doc.get('id'),
                    'filename': doc.get('filename'),
                    'error': str(e)
                })

        summary = {
            'status': 'completed',
            'total_documents': total_docs,
            'reindexed': reindexed_count,
            'failed': total_docs - reindexed_count,
            'results': results,
            'completed_at': datetime.now().isoformat()
        }

        logger.info(f"Document re-indexing completed: {reindexed_count}/{total_docs} successful")
        return summary

    except Exception as e:
        logger.error(f"Document re-indexing failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.document_tasks.cleanup_orphaned_documents')
def cleanup_orphaned_documents(self) -> Dict[str, Any]:
    """
    Clean up orphaned document indices and tracking records.
    Removes indices for documents that no longer exist on disk.

    Returns:
        Dictionary with cleanup results
    """
    try:
        logger.info("Starting orphaned document cleanup")

        # Get all tracked documents
        all_documents = asyncio.run(document_tracking_service.get_all_documents())

        logger.info(f"Found {len(all_documents)} tracked documents")

        total_docs = len(all_documents)
        cleaned_count = 0
        results = []

        for i, doc in enumerate(all_documents):
            try:
                # Update progress
                progress = int((i / total_docs) * 100)
                self.update_state(state='PROGRESS', meta={
                    'current': i + 1,
                    'total': total_docs,
                    'progress': progress,
                    'current_document': doc.get('filename', 'unknown')
                })

                file_path = doc.get('file_path')
                if not file_path or not os.path.exists(file_path):
                    # File doesn't exist, clean up tracking and index
                    asyncio.run(document_tracking_service.delete_document(doc['id']))

                    try:
                        asyncio.run(qdrant_service.delete_document_chunks(doc['filename'], doc['user_token']))
                    except Exception as index_error:
                        logger.warning(f"Failed to delete index for {doc['filename']}: {str(index_error)}")

                    # Invalidate caches
                    asyncio.run(cache_service.invalidate_document_data(doc['user_token'], doc['filename']))

                    cleaned_count += 1
                    results.append({
                        'action': 'cleaned',
                        'document_id': doc['id'],
                        'filename': doc['filename'],
                        'reason': 'file_not_found'
                    })
                else:
                    results.append({
                        'action': 'kept',
                        'document_id': doc['id'],
                        'filename': doc['filename']
                    })

            except Exception as e:
                logger.error(f"Error cleaning document {doc.get('filename')}: {str(e)}")
                results.append({
                    'action': 'error',
                    'document_id': doc.get('id'),
                    'filename': doc.get('filename'),
                    'error': str(e)
                })

        summary = {
            'status': 'completed',
            'total_documents': total_docs,
            'cleaned': cleaned_count,
            'kept': total_docs - cleaned_count,
            'results': results,
            'completed_at': datetime.now().isoformat()
        }

        logger.info(f"Orphaned document cleanup completed: {cleaned_count} documents cleaned")
        return summary

    except Exception as e:
        logger.error(f"Orphaned document cleanup failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.document_tasks.generate_document_stats')
def generate_document_stats(self, user_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive statistics about documents and their processing status.

    Args:
        user_token: Optional user token to filter stats

    Returns:
        Dictionary with document statistics
    """
    try:
        logger.info(f"Generating document statistics for user: {user_token or 'all'}")

        # Get documents
        if user_token:
            documents = asyncio.run(document_tracking_service.get_user_documents(user_token))
        else:
            documents = asyncio.run(document_tracking_service.get_all_documents())

        # Calculate statistics
        stats = {
            'total_documents': len(documents),
            'processed_documents': sum(1 for d in documents if d.get('status') == 'processed'),
            'failed_documents': sum(1 for d in documents if d.get('status') == 'failed'),
            'pending_documents': sum(1 for d in documents if d.get('status') == 'pending'),
            'total_chunks': sum(d.get('chunk_count', 0) for d in documents),
            'total_size_bytes': sum(d.get('file_size', 0) for d in documents),
            'avg_chunks_per_doc': 0,
            'avg_size_per_doc': 0,
            'documents_by_type': {},
            'processing_timeline': {},
            'generated_at': datetime.now().isoformat()
        }

        if stats['total_documents'] > 0:
            stats['avg_chunks_per_doc'] = stats['total_chunks'] / stats['total_documents']
            stats['avg_size_per_doc'] = stats['total_size_bytes'] / stats['total_documents']

        # Documents by file type
        for doc in documents:
            filename = doc.get('filename', '')
            if '.' in filename:
                ext = filename.split('.')[-1].lower()
                stats['documents_by_type'][ext] = stats['documents_by_type'].get(ext, 0) + 1

        # Processing timeline (by date)
        for doc in documents:
            timestamp = doc.get('ingestion_timestamp', '')
            if timestamp:
                date = timestamp.split('T')[0]  # YYYY-MM-DD
                stats['processing_timeline'][date] = stats['processing_timeline'].get(date, 0) + 1

        logger.info(f"Document statistics generated: {stats['total_documents']} documents")
        return stats

    except Exception as e:
        logger.error(f"Failed to generate document stats: {str(e)}")
        raise