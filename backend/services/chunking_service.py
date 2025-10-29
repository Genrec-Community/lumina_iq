from typing import List, Tuple, Dict, Any
import re
from utils.logger import chat_logger
from services.document_metadata_extractor import document_metadata_extractor


class ChunkingService:
    """Service for splitting text into chunks for RAG with rich metadata extraction"""

    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: The text to chunk
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text.strip():
            chat_logger.warning("Empty text provided for chunking")
            return []

        text = text.strip()
        if len(text) <= chunk_size:
            return [text]

        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size - 1

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                # Find the best break point
                best_break = max(
                    text.rfind("\n\n", start, end),
                    max(
                        (text.rfind(char, start, end) for char in [".", "!", "?"]),
                        default=-1,
                    ),
                )
                end = best_break + 2 if best_break != -1 else end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - chunk_overlap
            start = max(start, 0)  # Ensure start is not negative

        chat_logger.info(f"Chunking completed, created {len(chunks)} chunks")
        return chunks

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

    @staticmethod
    def chunk_with_rich_metadata(
        text: str, document_name: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Chunk text with rich structural metadata extraction.
        This is the ADVANCED method that extracts chapter, section, page, and content type info.

        Args:
            text: The full document text
            document_name: Name of the document
            chunk_size: Target chunk size
            overlap: Overlap between chunks

        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        chat_logger.info(
            "Chunking with rich metadata extraction",
            document_name=document_name,
            chunk_size=chunk_size,
        )

        # Create chunks
        chunks = ChunkingService.chunk_text(text, chunk_size, overlap)

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

        chat_logger.info(
            f"Created {len(chunks_with_metadata)} chunks with rich metadata"
        )

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


chunking_service = ChunkingService()
