from pathlib import Path
import sys
from typing import Iterator, List, Tuple, Dict, Any
import re
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.logger import chat_logger
from services.document_metadata_extractor import document_metadata_extractor

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    chat_logger.warning("LangChain not available, falling back to custom chunking")


class ChunkingService:
    """Service for splitting text into chunks for RAG using LangChain RecursiveCharacterTextSplitter"""

    def __init__(self):
        self.chunk_size = 256
        self.chunk_overlap = 0  # No overlap as specified
        self.separators = ["\n\n", "\n", " ", ""]  # Standard separators

        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                keep_separator=True
            )
        else:
            self.text_splitter = None

    async def chunk_text_async(self, text: str) -> List[str]:
        """
        Async method to split text into chunks using LangChain RecursiveCharacterTextSplitter

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        try:
            if not text or not text.strip():
                chat_logger.warning("Empty text provided for chunking")
                return []

            text = text.strip()

            if self.text_splitter:
                # Use LangChain splitter
                chunks = self.text_splitter.split_text(text)
                chat_logger.info(f"LangChain chunking completed, created {len(chunks)} chunks")
                return chunks
            else:
                # Fallback to custom chunking
                return await self._chunk_text_fallback(text)

        except Exception as e:
            chat_logger.error(f"Error in chunk_text_async: {str(e)}")
            raise

    async def _chunk_text_fallback(self, text: str) -> List[str]:
        """Fallback chunking method when LangChain is not available"""
        chunks = []
        chunk_size = self.chunk_size
        start = 0

        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                # Find natural break points
                break_pos = -1
                for sep in self.separators[:-1]:  # Skip empty separator
                    pos = text.rfind(sep, start, end)
                    if pos != -1:
                        break_pos = pos + len(sep)
                        break

                if break_pos == -1:
                    break_pos = end
                end = break_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end

        chat_logger.info(f"Fallback chunking completed, created {len(chunks)} chunks")
        return chunks

    # Keep legacy method for backward compatibility
    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> Iterator[str]:
        """
        Legacy sync method for backward compatibility
        """
        # Global instance for backward compatibility
        chunking_service = ChunkingService()
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(chunking_service.chunk_text_async(text))
            for chunk in result:
                yield chunk
        finally:
            loop.close()

    @staticmethod
    def chunk_by_paragraphs(text: str, max_chunk_size: int = 1500) -> List[str]:
        """
        Split text by paragraphs, combining small paragraphs to reach target size

        Args:
            text: The text to chunk
            max_chunk_size: Maximum size of each chunk

        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text.strip())

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds max size and we have content, save current chunk
            if len(current_chunk) + len(para) + 2 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        chat_logger.info(f"Split text into {len(chunks)} paragraph-based chunks")

        return chunks

    @staticmethod
    def chunk_with_context(
        text: str, chunk_size: int = 1000, overlap: int = 200, add_metadata: bool = True
    ) -> List[Tuple[str, dict]]:
        """
        Chunk text with additional context metadata

        Args:
            text: The text to chunk
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            add_metadata: Whether to add position metadata

        Returns:
            List of tuples (chunk_text, metadata_dict)
        """
        chunks = ChunkingService.chunk_text(text, chunk_size, overlap)

        if not add_metadata:
            return [(chunk, {}) for chunk in chunks]

        result = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "position": f"{i + 1}/{len(chunks)}",
                "is_first": i == 0,
                "is_last": i == len(chunks) - 1,
            }
            result.append((chunk, metadata))

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def chunk_with_rich_metadata_async(
        self, text: str, document_name: str
    ) -> List[Dict[str, Any]]:
        """
        Async method to chunk text with rich structural metadata extraction.
        This is the ADVANCED method that extracts chapter, section, page, and content type info.

        Args:
            text: The full document text
            document_name: Name of the document

        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        try:
            chat_logger.info("Chunking with rich metadata extraction")
            chat_logger.debug(f"Document name: {document_name}, text length: {len(text)} chars")

            # Create chunks using async method
            chunks = await self.chunk_text_async(text)

            if not chunks:
                return []

            # Extract metadata for each chunk
            chunks_with_metadata = []
            context_before = ""

            for i, chunk_text in enumerate(chunks):
                # Extract rich metadata
                metadata = document_metadata_extractor.extract_metadata_from_chunk(
                    chunk_text=chunk_text,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    document_name=document_name,
                    context_before=context_before if i > 0 else None,
                )

                chunks_with_metadata.append({"text": chunk_text, "metadata": metadata})

                # Update context for next iteration
                context_before = chunk_text

            # Propagate chapter/section metadata forward
            metadata_list = [c["metadata"] for c in chunks_with_metadata]
            updated_metadata = document_metadata_extractor.propagate_chapter_metadata(
                metadata_list
            )

            # Update with propagated metadata
            for i, chunk_data in enumerate(chunks_with_metadata):
                chunk_data["metadata"] = updated_metadata[i]

            chat_logger.info(f"Created {len(chunks_with_metadata)} chunks with rich metadata")

            # Log metadata statistics
            chapters_found = set(
                m.get("chapter_number") for m in updated_metadata if m.get("chapter_number")
            )
            sections_found = set(
                m.get("section_number") for m in updated_metadata if m.get("section_number")
            )

            chat_logger.info(
                f"Metadata stats: {len(chapters_found)} chapters, {len(sections_found)} sections found"
            )

            return chunks_with_metadata

        except Exception as e:
            chat_logger.error(f"Error in chunk_with_rich_metadata_async: {str(e)}")
            raise

    # Keep legacy method for backward compatibility
    @staticmethod
    def chunk_with_rich_metadata(
        text: str, document_name: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Legacy sync method for backward compatibility
        """
        chunking_service = ChunkingService()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                chunking_service.chunk_with_rich_metadata_async(text, document_name)
            )
        finally:
            loop.close()


chunking_service = ChunkingService()
