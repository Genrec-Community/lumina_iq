"""
Text chunking service using LangChain text splitters.
Provides intelligent text chunking for RAG pipeline.
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("chunking_service")


class ChunkingService:
    """Service for chunking text documents"""

    def __init__(self):
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.LLAMAINDEX_CHUNK_SIZE,
            chunk_overlap=settings.LLAMAINDEX_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with metadata.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dicts with text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        try:
            chunks = self._text_splitter.split_text(text)

            chunk_dicts = []
            for idx, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["chunk_count"] = len(chunks)

                chunk_dicts.append({
                    "text": chunk,
                    "metadata": chunk_metadata,
                })

            logger.info(
                f"Chunked text into {len(chunks)} pieces",
                extra={
                    "extra_fields": {
                        "original_length": len(text),
                        "chunk_count": len(chunks),
                    }
                },
            )

            return chunk_dicts

        except Exception as e:
            logger.error(
                f"Text chunking failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    def chunk_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dicts with 'text' and optional 'metadata'

        Returns:
            List of chunked documents
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks"
        )

        return all_chunks

    def create_custom_chunker(
        self, chunk_size: int, chunk_overlap: int
    ) -> RecursiveCharacterTextSplitter:
        """Create a custom text splitter with specific parameters"""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


# Global instance
chunking_service = ChunkingService()
