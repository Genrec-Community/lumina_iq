"""
RAG Orchestrator Service
High-level orchestration of RAG pipeline: ingestion → chunking → embedding → searching → generation
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import hashlib

from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.search_service import SearchService
from services.generation_service import GenerationService
from services.ingestion_service import IngestionService
from services.cache_service import cache_service
from services.celery_service import celery_service
from services.qdrant_service import qdrant_service
from services.document_tracking_service import DocumentTrackingService

from utils.logger import get_logger
from config.settings import settings

logger = get_logger("rag_orchestrator")


class RAGOrchestrator:
    """
    High-level RAG orchestrator that coordinates the entire RAG pipeline.
    Handles caching, fallbacks, circuit breakers, and background processing.
    """

    def __init__(self):
        self.chunking_service = ChunkingService()
        self.embedding_service = EmbeddingService()
        self.search_service = SearchService()
        self.generation_service = GenerationService()
        self.ingestion_service = IngestionService()
        self.document_tracker = DocumentTrackingService()

        # Circuit breaker state
        self.circuit_breaker_failures = {}
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.failure_threshold = 3

    async def _is_circuit_breaker_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service"""
        if service_name not in self.circuit_breaker_failures:
            return False

        failures = self.circuit_breaker_failures[service_name]
        if len(failures) < self.failure_threshold:
            return False

        # Check if timeout has passed
        last_failure = max(failures)
        if (datetime.now().timestamp() - last_failure) > self.circuit_breaker_timeout:
            # Reset circuit breaker
            self.circuit_breaker_failures[service_name] = []
            return False

        return True

    def _record_service_failure(self, service_name: str):
        """Record a service failure for circuit breaker"""
        if service_name not in self.circuit_breaker_failures:
            self.circuit_breaker_failures[service_name] = []

        self.circuit_breaker_failures[service_name].append(datetime.now().timestamp())

        # Keep only recent failures
        cutoff = datetime.now().timestamp() - self.circuit_breaker_timeout
        self.circuit_breaker_failures[service_name] = [
            f for f in self.circuit_breaker_failures[service_name] if f > cutoff
        ]

    def _record_service_success(self, service_name: str):
        """Record a service success to reset circuit breaker"""
        if service_name in self.circuit_breaker_failures:
            # Clear failures on success
            self.circuit_breaker_failures[service_name] = []

    async def _safe_service_call(self, service_name: str, service_call, *args, **kwargs):
        """Execute a service call with circuit breaker protection"""
        if self._is_circuit_breaker_open(service_name):
            logger.warning(f"Circuit breaker open for {service_name}, skipping call")
            return None

        try:
            result = await service_call(*args, **kwargs)
            self._record_service_success(service_name)
            return result
        except Exception as e:
            logger.error(f"Service call failed for {service_name}: {str(e)}")
            self._record_service_failure(service_name)
            raise

    async def _get_cached_result(self, cache_key: str, cache_type: str = "api") -> Optional[Any]:
        """Get cached result with fallback"""
        try:
            if cache_type == "api":
                return await cache_service.get_api_response(cache_key)
            elif cache_type == "retrieval":
                # For retrieval results, we'd need token and filename
                return None
            elif cache_type == "embedding":
                return await cache_service.get_embedding(cache_key)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
            return None

    async def _set_cached_result(self, cache_key: str, result: Any, ttl: int = None, cache_type: str = "api"):
        """Cache result with error handling"""
        try:
            if cache_type == "api":
                await cache_service.set_api_response(cache_key, result, ttl)
            elif cache_type == "retrieval":
                # For retrieval results, we'd need token and filename
                pass
            elif cache_type == "embedding":
                await cache_service.set_embedding(cache_key, result)
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")

    def _generate_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate consistent cache key"""
        key_data = f"{operation}:{sorted(params.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def process_document_ingestion(self, file_path: str, user_token: str) -> Dict[str, Any]:
        """
        Orchestrate document ingestion pipeline.
        For large files, delegate to Celery background processing.
        """

        # Check if document already exists
        file_hash = await self.ingestion_service._get_file_hash(file_path)
        existing_doc = self.document_tracker.check_document_exists(user_token, file_hash)

        if existing_doc:
            logger.info(f"Document already exists: {existing_doc['filename']}")
            return {
                "status": "exists",
                "document_id": existing_doc["id"],
                "message": "Document already processed"
            }

        # Check file size for background processing
        file_size_mb = await self.ingestion_service._get_file_size_mb(file_path)

        if file_size_mb > 50:  # Large file threshold
            # Delegate to Celery
            task = celery_service.celery_app.send_task(
                'document_ingestion.process_large_document',
                args=[file_path, user_token, file_hash],
                queue='ingestion'
            )

            return {
                "status": "processing",
                "task_id": task.id,
                "message": f"Large document ({file_size_mb:.1f}MB) queued for processing"
            }

        # Process synchronously for smaller files
        return await self._process_document_pipeline(file_path, user_token, file_hash)

    async def _process_document_pipeline(self, file_path: str, user_token: str, file_hash: str) -> Dict[str, Any]:
        """Execute the complete document processing pipeline"""

        try:
            # Step 1: Load and validate document
            document = await self._safe_service_call(
                "ingestion",
                self.ingestion_service.load_document_async,
                file_path, user_token
            )

            # Step 2: Extract text content
            content = await self._safe_service_call(
                "ingestion",
                self.ingestion_service.process_document_async,
                file_path, user_token
            )

            # Step 3: Chunk the content
            chunks = await self._safe_service_call(
                "chunking",
                self.chunking_service.chunk_text_async,
                content["text"]
            )

            # Step 4: Generate embeddings for chunks
            embeddings = await self._safe_service_call(
                "embedding",
                self.embedding_service.generate_embeddings_batch,
                chunks
            )

            # Step 5: Index in vector store
            await self._safe_service_call(
                "qdrant",
                qdrant_service.index_document,
                chunks, embeddings, content["metadata"], user_token
            )

            # Step 6: Track document
            doc_record = self.document_tracker.add_document(
                user_id=user_token,
                file_hash=file_hash,
                filename=content["metadata"]["filename"],
                chunk_count=len(chunks),
                file_size_bytes=content["metadata"]["file_size"]
            )

            logger.info(f"Document processing completed: {content['metadata']['filename']}")

            return {
                "status": "success",
                "document_id": doc_record["id"],
                "chunks_processed": len(chunks),
                "message": f"Document processed successfully with {len(chunks)} chunks"
            }

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")

            # Cleanup on failure
            try:
                await qdrant_service.delete_document_chunks(content["metadata"]["filename"], user_token)
            except:
                pass

            return {
                "status": "error",
                "message": f"Document processing failed: {str(e)}"
            }

    async def retrieve_and_generate_questions(
        self,
        query: str,
        token: str,
        filename: str,
        count: int = 25,
        mode: str = "practice"
    ) -> Dict[str, Any]:
        """
        Main orchestration method for question generation.
        Coordinates retrieval, caching, and generation with fallbacks.
        """

        # Generate cache key
        cache_key = self._generate_cache_key("generate_questions", {
            "query": query,
            "filename": filename,
            "count": count,
            "mode": mode
        })

        # Check cache first
        cached_result = await self._get_cached_result(cache_key, "api")
        if cached_result:
            logger.info("Returning cached question generation result")
            return cached_result

        try:
            # Step 1: Retrieve relevant context
            retrieval_result = await self._perform_retrieval(query, token, filename)

            if retrieval_result["status"] != "success":
                return {
                    "status": "error",
                    "message": "Failed to retrieve context for question generation"
                }

            # Step 2: Generate questions
            generation_result = await self._perform_question_generation(
                retrieval_result["context"],
                query,
                count,
                mode
            )

            if generation_result["status"] != "success":
                return {
                    "status": "error",
                    "message": "Failed to generate questions"
                }

            # Step 3: Cache successful result
            result = {
                "status": "success",
                "response": generation_result["response"],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "chunks_used": len(retrieval_result.get("chunks", [])),
                    "retrieval_method": retrieval_result.get("retrieval_method", "unknown"),
                    "generation_mode": mode
                }
            }

            await self._set_cached_result(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)

            return result

        except Exception as e:
            logger.error(f"Question generation orchestration failed: {str(e)}")

            # Return fallback response
            return {
                "status": "error",
                "message": f"Question generation failed: {str(e)}",
                "response": "I'm sorry, but I encountered an error while generating questions. Please try again.",
                "timestamp": datetime.now().isoformat()
            }

    async def _perform_retrieval(self, query: str, token: str, filename: str) -> Dict[str, Any]:
        """Perform context retrieval with multiple strategies and fallbacks"""

        # Strategy 1: Try advanced RAG service first
        try:
            from services.advanced_rag_service import advanced_rag_service

            result = await advanced_rag_service.retrieve_for_questions(
                query=query,
                token=token,
                filename=filename,
                num_questions=25  # This affects retrieval breadth
            )

            if result["status"] == "success" and result["context"]:
                logger.info("Using advanced RAG retrieval")
                return result

        except Exception as e:
            logger.warning(f"Advanced RAG failed, trying fallback: {str(e)}")

        # Strategy 2: Try Q&A generation service with HyDE
        try:
            from services.qa_generation_service import qa_generation_service

            result = await qa_generation_service.hyde_retrieval(
                query=query,
                token=token,
                filename=filename,
                num_results=20
            )

            if result["status"] == "success" and result["enhanced_context"]:
                logger.info("Using Q&A generation service retrieval")
                return {
                    "status": "success",
                    "context": result["enhanced_context"],
                    "chunks": result.get("chunks", []),
                    "retrieval_method": "hyde_rag"
                }

        except Exception as e:
            logger.warning(f"Q&A generation service failed, trying basic retrieval: {str(e)}")

        # Strategy 3: Basic RAG service fallback
        try:
            from services.rag_service import rag_service

            result = await rag_service.retrieve_context(
                query=query,
                token=token,
                filename=filename,
                top_k=15
            )

            if result["status"] == "success":
                logger.info("Using basic RAG retrieval")
                return result

        except Exception as e:
            logger.warning(f"Basic RAG failed: {str(e)}")

        # Strategy 4: Direct search service
        try:
            result = await self.search_service.search_similar_chunks(
                query=query,
                token=token,
                filename=filename,
                limit=15
            )

            if result["status"] == "success" and result["chunks"]:
                context_parts = [chunk["text"] for chunk in result["chunks"]]
                combined_context = "\n\n".join(context_parts)

                logger.info("Using direct search service retrieval")
                return {
                    "status": "success",
                    "context": combined_context,
                    "chunks": result["chunks"],
                    "retrieval_method": "search_service"
                }

        except Exception as e:
            logger.error(f"All retrieval strategies failed: {str(e)}")

        # Final fallback: return error
        return {
            "status": "error",
            "message": "All retrieval methods failed",
            "context": "",
            "chunks": []
        }

    async def _perform_question_generation(
        self,
        context: str,
        query: str,
        count: int,
        mode: str
    ) -> Dict[str, Any]:
        """Generate questions using the generation service"""

        try:
            result = await self.generation_service.generate_questions(
                context=context,
                count=count,
                mode=mode,
                topic=query if query else None
            )

            if result["status"] == "success":
                return result

        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")

        # Fallback: return error
        return {
            "status": "error",
            "message": f"Question generation failed: {str(e)}"
        }

    async def get_orchestrator_health(self) -> Dict[str, Any]:
        """Get health status of all orchestrated services"""

        services_status = {}

        # Check cache service
        try:
            cache_health = await cache_service.get_cache_health()
            services_status["cache"] = cache_health
        except Exception as e:
            services_status["cache"] = {"status": "error", "error": str(e)}

        # Check Celery service
        try:
            celery_health = celery_service.health_check()
            services_status["celery"] = celery_health
        except Exception as e:
            services_status["celery"] = {"status": "error", "error": str(e)}

        # Check circuit breakers
        services_status["circuit_breakers"] = {
            service: "open" if self._is_circuit_breaker_open(service) else "closed"
            for service in ["ingestion", "chunking", "embedding", "qdrant", "generation"]
        }

        # Overall status
        critical_services = ["cache", "celery"]
        overall_healthy = all(
            services_status.get(service, {}).get("status") == "healthy"
            for service in critical_services
        )

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "services": services_status,
            "timestamp": datetime.now().isoformat()
        }


# Global orchestrator instance
rag_orchestrator = RAGOrchestrator()