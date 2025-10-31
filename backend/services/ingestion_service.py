"""
Document ingestion service for processing and indexing documents.
"""

from typing import Dict, Any, Optional
from services.chunking_service import chunking_service
from services.qdrant_service import qdrant_service
from utils.logger import get_logger
import datetime

logger = get_logger("ingestion_service")


class IngestionService:
    """Service for document ingestion and indexing"""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Initialize the ingestion service"""
        if not self._initialized:
            qdrant_service.initialize()
            self._initialized = True

    async def ingest_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document into the system.

        Args:
            text: Document text
            metadata: Optional metadata
            chunk_size: Optional custom chunk size
            chunk_overlap: Optional custom chunk overlap

        Returns:
            Ingestion result
        """
        if not self._initialized:
            self.initialize()

        try:
            # Add ingestion timestamp
            if metadata is None:
                metadata = {}

            metadata["ingestion_time"] = datetime.datetime.utcnow().isoformat()

            # Chunk the document
            if chunk_size and chunk_overlap:
                chunker = chunking_service.create_custom_chunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                chunks = chunker.split_text(text)
                chunk_dicts = [
                    {
                        "text": chunk,
                        "metadata": {**metadata, "chunk_index": idx, "chunk_count": len(chunks)},
                    }
                    for idx, chunk in enumerate(chunks)
                ]
            else:
                chunk_dicts = chunking_service.chunk_text(text, metadata)

            # Extract texts and metadata
            chunk_texts = [chunk["text"] for chunk in chunk_dicts]
            chunk_metadata = [chunk["metadata"] for chunk in chunk_dicts]

            # Add to vector store
            doc_ids = await qdrant_service.add_documents(
                texts=chunk_texts,
                metadata=chunk_metadata,
            )

            logger.info(
                f"Document ingested successfully: {len(chunk_dicts)} chunks",
                extra={
                    "extra_fields": {
                        "chunk_count": len(chunk_dicts),
                        "metadata": metadata,
                    }
                },
            )

            return {
                "status": "success",
                "chunk_count": len(chunk_dicts),
                "document_ids": doc_ids,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(
                f"Document ingestion failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {
                "status": "error",
                "message": str(e),
            }

    async def batch_ingest_documents(
        self,
        documents: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents in batch.

        Args:
            documents: List of document dicts with 'text' and optional 'metadata'

        Returns:
            Batch ingestion result
        """
        if not self._initialized:
            self.initialize()

        results = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            result = await self.ingest_document(text, metadata)
            results.append(result)

        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful

        logger.info(
            f"Batch ingestion completed: {successful} successful, {failed} failed",
        )

        return {
            "status": "completed",
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
        }


# Global instance
ingestion_service = IngestionService()
