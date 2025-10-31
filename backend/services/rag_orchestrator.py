"""
RAG orchestrator service for coordinating RAG operations.
Manages document ingestion, querying, and question generation.
"""

from typing import Dict, Any, Optional
from services.rag_service import rag_service
from services.qdrant_service import qdrant_service
from utils.logger import get_logger
from utils.storage import pdf_contexts, storage_manager
import datetime

logger = get_logger("rag_orchestrator")


class RAGOrchestrator:
    """Orchestrates RAG operations and workflows"""

    def __init__(self):
        self._initialized = False
        self._ingested_documents = set()

    def initialize(self):
        """Initialize the RAG orchestrator"""
        if not self._initialized:
            rag_service.initialize()
            self._initialized = True

    async def ingest_pdf_for_user(
        self,
        user_session: str,
        force_reingest: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest PDF for a specific user session.

        Args:
            user_session: User session ID
            force_reingest: Force re-ingestion even if already ingested

        Returns:
            Ingestion result
        """
        if not self._initialized:
            self.initialize()

        try:
            # Get PDF context
            pdf_context = storage_manager.safe_get(pdf_contexts, user_session)

            if not pdf_context:
                return {
                    "status": "error",
                    "message": "No PDF selected for this session",
                }

            filename = pdf_context.get("filename")
            text = pdf_context.get("text")

            if not text:
                return {
                    "status": "error",
                    "message": "No text found in PDF context",
                }

            # Check if already ingested
            if filename in self._ingested_documents and not force_reingest:
                logger.info(f"Document already ingested: {filename}")
                return {
                    "status": "success",
                    "message": "Document already ingested",
                    "filename": filename,
                }

            # Ingest the document
            metadata = {
                "filename": filename,
                "user_session": user_session,
                "ingestion_time": datetime.datetime.utcnow().isoformat(),
            }

            result = await rag_service.ingest_document(text, metadata)

            if result["status"] == "success":
                self._ingested_documents.add(filename)

            return result

        except Exception as e:
            logger.error(
                f"PDF ingestion failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {
                "status": "error",
                "message": str(e),
            }

    async def retrieve_and_answer(
        self,
        query: str,
        token: str,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents and generate answer.

        Args:
            query: User query
            token: User session token
            filename: Optional filename filter

        Returns:
            Answer with retrieved documents
        """
        if not self._initialized:
            self.initialize()

        try:
            # Build filters
            filters = {}
            if filename:
                filters["filename"] = filename

            # Query the RAG system
            result = await rag_service.query(
                query_text=query,
                top_k=10,
                score_threshold=0.7,
                filters=filters if filters else None,
            )

            return result

        except Exception as e:
            logger.error(
                f"Retrieve and answer failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {
                "status": "error",
                "message": str(e),
            }

    async def retrieve_and_generate_questions(
        self,
        query: str,
        token: str,
        filename: str,
        count: int = 25,
        mode: str = "practice",
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context and generate questions.

        Args:
            query: Topic or query for questions
            token: User session token
            filename: Filename to filter by
            count: Number of questions
            mode: Question mode (quiz or practice)

        Returns:
            Generated questions
        """
        if not self._initialized:
            self.initialize()

        try:
            # Get PDF context for the user
            pdf_context = storage_manager.safe_get(pdf_contexts, token)

            if not pdf_context or pdf_context.get("filename") != filename:
                return {
                    "status": "error",
                    "message": "PDF context not found or filename mismatch",
                }

            text = pdf_context.get("text", "")

            if not text:
                return {
                    "status": "error",
                    "message": "No text found in PDF",
                }

            # Generate questions from the text
            questions = await rag_service.generate_questions(
                context=text,
                count=count,
                mode=mode,
            )

            return {
                "status": "success",
                "response": questions,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Question generation failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {
                "status": "error",
                "message": str(e),
            }

    async def get_orchestrator_health(self) -> Dict[str, Any]:
        """Get health status of the orchestrator"""
        try:
            collection_info = await qdrant_service.get_collection_info()

            return {
                "status": "operational",
                "initialized": self._initialized,
                "ingested_documents": len(self._ingested_documents),
                "vector_store": collection_info,
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
            }


# Global instance
rag_orchestrator = RAGOrchestrator()
