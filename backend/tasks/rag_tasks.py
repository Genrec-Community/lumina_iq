"""
Celery tasks for RAG-specific background operations.
Handles batch processing, index optimization, and RAG maintenance tasks.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .celery_app import app
from backend.services.rag_service import rag_service
from backend.services.qdrant_service import qdrant_service
from backend.services.embedding_service import embedding_service
from backend.services.cache_service import cache_service
from backend.services.retrieval_strategy_manager import retrieval_strategy_manager
from backend.services.document_tracking_service import document_tracking_service
from utils.logger import get_logger

logger = get_logger("rag_tasks")


@app.task(bind=True, name='tasks.rag_tasks.batch_index_documents')
def batch_index_documents(self, documents: List[Dict[str, Any]], user_token: str) -> Dict[str, Any]:
    """
    Batch index multiple documents for RAG retrieval.
    Optimized for bulk operations with progress tracking.

    Args:
        documents: List of document dictionaries with 'filename', 'content', 'file_path'
        user_token: User session token

    Returns:
        Dictionary with batch indexing results
    """
    try:
        logger.info(f"Starting batch indexing for {len(documents)} documents")

        total_docs = len(documents)
        indexed_count = 0
        failed_count = 0
        results = []

        for i, doc_data in enumerate(documents):
            try:
                # Update progress
                progress = int((i / total_docs) * 100)
                self.update_state(state='PROGRESS', meta={
                    'current': i + 1,
                    'total': total_docs,
                    'progress': progress,
                    'current_document': doc_data.get('filename', 'unknown')
                })

                # Index individual document
                result = asyncio.run(rag_service.index_document(
                    filename=doc_data['filename'],
                    content=doc_data['content'],
                    token=user_token,
                    file_path=doc_data.get('file_path')
                ))

                if result.get('status') in ['success', 'already_indexed']:
                    indexed_count += 1
                else:
                    failed_count += 1

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to index document {doc_data.get('filename')}: {str(e)}")
                failed_count += 1
                results.append({
                    'status': 'error',
                    'filename': doc_data.get('filename'),
                    'error': str(e)
                })

        # Invalidate relevant caches after batch operation
        asyncio.run(cache_service.invalidate_user_data(user_token))

        summary = {
            'status': 'completed',
            'total_documents': total_docs,
            'indexed': indexed_count,
            'failed': failed_count,
            'results': results,
            'user_token': user_token,
            'completed_at': datetime.now().isoformat()
        }

        logger.info(f"Batch indexing completed: {indexed_count}/{total_docs} successful")
        return summary

    except Exception as e:
        logger.error(f"Batch indexing failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.rag_tasks.optimize_vector_index')
def optimize_vector_index(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Optimize vector database index for better performance.
    Performs index optimization, cleanup, and maintenance operations.

    Args:
        collection_name: Optional specific collection to optimize

    Returns:
        Dictionary with optimization results
    """
    try:
        logger.info(f"Starting vector index optimization for collection: {collection_name or 'all'}")

        # Get collections to optimize
        if collection_name:
            collections = [collection_name]
        else:
            collections = asyncio.run(qdrant_service.list_collections())

        logger.info(f"Optimizing {len(collections)} collections")

        optimization_results = {}

        for i, collection in enumerate(collections):
            try:
                # Update progress
                progress = int((i / len(collections)) * 100)
                self.update_state(state='PROGRESS', meta={
                    'current': i + 1,
                    'total': len(collections),
                    'progress': progress,
                    'current_collection': collection
                })

                # Perform index optimization (simplified)
                # In practice, this would involve Qdrant-specific optimization commands
                optimize_result = asyncio.run(qdrant_service.optimize_collection(collection))

                optimization_results[collection] = optimize_result

            except Exception as e:
                logger.error(f"Failed to optimize collection {collection}: {str(e)}")
                optimization_results[collection] = {'error': str(e)}

        summary = {
            'status': 'completed',
            'collections_optimized': len(collections),
            'optimization_results': optimization_results,
            'optimized_at': datetime.now().isoformat()
        }

        logger.info(f"Vector index optimization completed for {len(collections)} collections")
        return summary

    except Exception as e:
        logger.error(f"Vector index optimization failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.rag_tasks.update_embeddings_batch')
def update_embeddings_batch(self, document_ids: List[str], user_token: str) -> Dict[str, Any]:
    """
    Update embeddings for existing documents after model changes.
    Useful when switching embedding models or updating configurations.

    Args:
        document_ids: List of document IDs to update embeddings for
        user_token: User session token

    Returns:
        Dictionary with update results
    """
    try:
        logger.info(f"Starting batch embedding update for {len(document_ids)} documents")

        total_docs = len(document_ids)
        updated_count = 0
        failed_count = 0
        results = []

        for i, doc_id in enumerate(document_ids):
            try:
                # Update progress
                progress = int((i / total_docs) * 100)
                self.update_state(state='PROGRESS', meta={
                    'current': i + 1,
                    'total': total_docs,
                    'progress': progress,
                    'current_document_id': doc_id
                })

                # Get document data
                doc = asyncio.run(document_tracking_service.get_document_by_id(doc_id))
                if not doc:
                    logger.warning(f"Document not found: {doc_id}")
                    failed_count += 1
                    results.append({'document_id': doc_id, 'status': 'not_found'})
                    continue

                # Delete existing embeddings
                asyncio.run(qdrant_service.delete_document_chunks(doc['filename'], user_token))

                # Re-index document (this will generate new embeddings)
                with open(doc['file_path'], 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                index_result = asyncio.run(rag_service.index_document(
                    filename=doc['filename'],
                    content=content,
                    token=user_token,
                    file_path=doc['file_path']
                ))

                if index_result.get('status') in ['success', 'already_indexed']:
                    updated_count += 1
                    results.append({
                        'document_id': doc_id,
                        'status': 'updated',
                        'index_result': index_result
                    })
                else:
                    failed_count += 1
                    results.append({
                        'document_id': doc_id,
                        'status': 'failed',
                        'error': index_result
                    })

            except Exception as e:
                logger.error(f"Failed to update embeddings for document {doc_id}: {str(e)}")
                failed_count += 1
                results.append({
                    'document_id': doc_id,
                    'status': 'error',
                    'error': str(e)
                })

        # Clear embedding caches
        asyncio.run(cache_service.invalidate_user_data(user_token))

        summary = {
            'status': 'completed',
            'total_documents': total_docs,
            'updated': updated_count,
            'failed': failed_count,
            'results': results,
            'user_token': user_token,
            'completed_at': datetime.now().isoformat()
        }

        logger.info(f"Batch embedding update completed: {updated_count}/{total_docs} successful")
        return summary

    except Exception as e:
        logger.error(f"Batch embedding update failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.rag_tasks.validate_rag_system')
def validate_rag_system(self) -> Dict[str, Any]:
    """
    Validate RAG system components and performance.
    Performs comprehensive checks on retrieval, embedding, and indexing systems.

    Returns:
        Dictionary with validation results
    """
    try:
        logger.info("Starting RAG system validation")

        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'performance_tests': {},
            'recommendations': []
        }

        # Test Qdrant connectivity and health
        try:
            collections = asyncio.run(qdrant_service.list_collections())
            validation_results['components']['qdrant'] = {
                'status': 'healthy',
                'collections_count': len(collections),
                'collections': collections
            }
        except Exception as e:
            validation_results['components']['qdrant'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            validation_results['recommendations'].append("Fix Qdrant connectivity issues")

        # Test embedding service
        try:
            test_embedding = asyncio.run(embedding_service.generate_embeddings_batch(["test query"]))
            validation_results['components']['embedding_service'] = {
                'status': 'healthy',
                'test_embedding_shape': len(test_embedding[0]) if test_embedding else 0
            }
        except Exception as e:
            validation_results['components']['embedding_service'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            validation_results['recommendations'].append("Fix embedding service issues")

        # Test retrieval performance
        try:
            test_retrieval = asyncio.run(retrieval_strategy_manager.retrieve(
                query="test query for validation",
                token="system_test",
                top_k=3
            ))
            validation_results['performance_tests']['retrieval'] = {
                'status': 'passed',
                'results_count': test_retrieval.get('num_chunks', 0)
            }
        except Exception as e:
            validation_results['performance_tests']['retrieval'] = {
                'status': 'failed',
                'error': str(e)
            }
            validation_results['recommendations'].append("Fix retrieval system issues")

        # Overall system status
        component_statuses = [comp['status'] for comp in validation_results['components'].values()]
        validation_results['overall_status'] = 'healthy' if all(s == 'healthy' for s in component_statuses) else 'degraded'

        logger.info(f"RAG system validation completed: {validation_results['overall_status']}")
        return validation_results

    except Exception as e:
        logger.error(f"RAG system validation failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.rag_tasks.generate_rag_report')
def generate_rag_report(self, user_token: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
    """
    Generate comprehensive RAG system usage and performance report.

    Args:
        user_token: Optional user token to filter report
        days: Number of days to analyze

    Returns:
        Dictionary with RAG system report
    """
    try:
        logger.info(f"Generating RAG report for user: {user_token or 'all'} over {days} days")

        # Get document statistics
        if user_token:
            documents = asyncio.run(document_tracking_service.get_user_documents(user_token))
        else:
            documents = asyncio.run(document_tracking_service.get_all_documents())

        # Calculate RAG metrics
        rag_metrics = {
            'total_documents': len(documents),
            'indexed_documents': sum(1 for d in documents if d.get('status') == 'processed'),
            'total_chunks': sum(d.get('chunk_count', 0) for d in documents),
            'avg_chunks_per_doc': 0,
            'document_types': {},
            'indexing_efficiency': {},
            'retrieval_stats': {}
        }

        if rag_metrics['total_documents'] > 0:
            rag_metrics['avg_chunks_per_doc'] = rag_metrics['total_chunks'] / rag_metrics['total_documents']

        # Document type distribution
        for doc in documents:
            filename = doc.get('filename', '')
            if '.' in filename:
                ext = filename.split('.')[-1].lower()
                rag_metrics['document_types'][ext] = rag_metrics['document_types'].get(ext, 0) + 1

        # Vector store statistics
        try:
            collections_info = asyncio.run(qdrant_service.get_collections_info())
            rag_metrics['vector_store'] = collections_info
        except Exception as e:
            logger.warning(f"Failed to get vector store info: {str(e)}")
            rag_metrics['vector_store'] = {'error': str(e)}

        # Cache performance
        try:
            cache_stats = asyncio.run(cache_service.get_cache_stats())
            rag_metrics['cache_performance'] = cache_stats
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {str(e)}")
            rag_metrics['cache_performance'] = {'error': str(e)}

        # Generate recommendations
        recommendations = []

        if rag_metrics['total_documents'] > 0:
            indexing_rate = rag_metrics['indexed_documents'] / rag_metrics['total_documents']
            if indexing_rate < 0.8:
                recommendations.append(".2f")

        if rag_metrics.get('total_chunks', 0) > 100000:
            recommendations.append("Consider optimizing chunk size for better retrieval")

        report = {
            'status': 'completed',
            'user_token': user_token,
            'period_days': days,
            'rag_metrics': rag_metrics,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }

        logger.info("RAG report generated successfully")
        return report

    except Exception as e:
        logger.error(f"RAG report generation failed: {str(e)}")
        raise


@app.task(bind=True, name='tasks.rag_tasks.cleanup_rag_data')
def cleanup_rag_data(self, user_token: Optional[str] = None, older_than_days: int = 90) -> Dict[str, Any]:
    """
    Clean up old or orphaned RAG data.
    Removes outdated vectors, embeddings, and cached data.

    Args:
        user_token: Optional user token to clean data for specific user
        older_than_days: Remove data older than this many days

    Returns:
        Dictionary with cleanup results
    """
    try:
        logger.info(f"Starting RAG data cleanup for user: {user_token or 'all'}, older than {older_than_days} days")

        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=older_than_days)

        # Get documents to potentially clean
        if user_token:
            documents = asyncio.run(document_tracking_service.get_user_documents(user_token))
        else:
            documents = asyncio.run(document_tracking_service.get_all_documents())

        logger.info(f"Found {len(documents)} documents to evaluate for cleanup")

        cleaned_count = 0
        kept_count = 0
        results = []

        for i, doc in enumerate(documents):
            try:
                # Update progress
                progress = int((i / len(documents)) * 100)
                self.update_state(state='PROGRESS', meta={
                    'current': i + 1,
                    'total': len(documents),
                    'progress': progress,
                    'current_document': doc.get('filename', 'unknown')
                })

                # Check if document is old enough for cleanup
                ingestion_date = doc.get('ingestion_timestamp')
                if not ingestion_date:
                    kept_count += 1
                    results.append({
                        'document_id': doc.get('id'),
                        'action': 'kept',
                        'reason': 'no_ingestion_date'
                    })
                    continue

                try:
                    doc_date = datetime.fromisoformat(ingestion_date.replace('Z', '+00:00'))
                except:
                    kept_count += 1
                    results.append({
                        'document_id': doc.get('id'),
                        'action': 'kept',
                        'reason': 'invalid_date_format'
                    })
                    continue

                if doc_date > cutoff_date:
                    kept_count += 1
                    results.append({
                        'document_id': doc.get('id'),
                        'action': 'kept',
                        'reason': 'too_recent'
                    })
                    continue

                # Clean up document data
                # Delete from vector store
                asyncio.run(qdrant_service.delete_document_chunks(doc['filename'], doc.get('user_token', user_token)))

                # Delete from tracking
                asyncio.run(document_tracking_service.delete_document(doc['id']))

                # Invalidate caches
                asyncio.run(cache_service.invalidate_document_data(doc.get('user_token', user_token), doc['filename']))

                cleaned_count += 1
                results.append({
                    'document_id': doc.get('id'),
                    'action': 'cleaned',
                    'filename': doc['filename'],
                    'ingestion_date': ingestion_date
                })

            except Exception as e:
                logger.error(f"Error cleaning document {doc.get('filename')}: {str(e)}")
                results.append({
                    'document_id': doc.get('id'),
                    'action': 'error',
                    'filename': doc.get('filename'),
                    'error': str(e)
                })

        summary = {
            'status': 'completed',
            'user_token': user_token,
            'older_than_days': older_than_days,
            'total_documents': len(documents),
            'cleaned': cleaned_count,
            'kept': kept_count,
            'results': results,
            'completed_at': datetime.now().isoformat()
        }

        logger.info(f"RAG data cleanup completed: {cleaned_count} documents cleaned")
        return summary

    except Exception as e:
        logger.error(f"RAG data cleanup failed: {str(e)}")
        raise