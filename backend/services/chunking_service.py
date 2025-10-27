from typing import List, Tuple, Dict, Any
import re
from utils.logger import chat_logger
from services.document_metadata_extractor import document_metadata_extractor

class ChunkingService:
    """Service for splitting text into chunks for RAG with rich metadata extraction"""
    
    @staticmethod
    def chunk_text(
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
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
        if not text or len(text.strip()) == 0:
            chat_logger.warning("Empty text provided for chunking")
            return []
        
        # Clean the text
        text = text.strip()
        
        # If text is smaller than chunk_size, return as single chunk
        if len(text) <= chunk_size:
            return [text]
        
        # Prevent infinite loop if overlap is too large
        if chunk_overlap >= chunk_size:
            chat_logger.warning(f"Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}), reducing overlap to {chunk_size - 1}")
            chunk_overlap = chunk_size - 1
        
        chunks = []
        start = 0
        iteration_count = 0

        while start < len(text):
            # Calculate end position
            end = start + chunk_size
            iteration_count += 1
            if iteration_count % 100 == 0:
                chat_logger.info(f"Chunking iteration {iteration_count}, start={start}, end={end}, text_len={len(text)}")
            
            # If we're not at the end of the text, try to break at a sentence or paragraph
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start:
                    end = paragraph_break + 2
                    chat_logger.debug(f"Adjusted end to paragraph break at {paragraph_break}")
                else:
                    # Look for sentence break
                    sentence_breaks = ['.', '!', '?', '\n']
                    best_break = -1
                    for break_char in sentence_breaks:
                        pos = text.rfind(break_char, start, end)
                        if pos > best_break and pos > start:
                            best_break = pos
                    
                    if best_break != -1:
                        end = best_break + 1
                        chat_logger.debug(f"Adjusted end to sentence break at {best_break}")
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Track previous end to ensure progress
            prev_end = end
            
            # Move start position with overlap
            start = end - chunk_overlap
            
            # Ensure we make progress (don't go backward or stay in same position)
            if start < 0:
                start = 0
            if start <= prev_end:
                start = prev_end + 1
                chat_logger.debug(f"Adjusted start from {start - (prev_end + 1)} to {start} to ensure progress")
        
        chat_logger.info(f"Chunking completed after {iteration_count} iterations")
        if iteration_count > 1000:
            chat_logger.warning(f"High iteration count: {iteration_count}, possible infinite loop or performance issue")
        chat_logger.info(f"Split text into {len(chunks)} chunks",
                        total_length=len(text),
                        avg_chunk_size=len(text)//len(chunks) if chunks else 0)

        return chunks
    
    @staticmethod
    def chunk_by_paragraphs(
        text: str, 
        max_chunk_size: int = 1500
    ) -> List[str]:
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
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
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
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        add_metadata: bool = True
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
                "position": f"{i+1}/{len(chunks)}",
                "is_first": i == 0,
                "is_last": i == len(chunks) - 1
            }
            result.append((chunk, metadata))
        
        return result
    
    @staticmethod
    def chunk_with_rich_metadata(
        text: str,
        document_name: str,
        chunk_size: int = 1000,
        overlap: int = 200
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
        chat_logger.info("Chunking with rich metadata extraction", 
                        document_name=document_name,
                        chunk_size=chunk_size)
        
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
                context_before=context_before if i > 0 else None
            )
            
            chunks_with_metadata.append({
                "text": chunk_text,
                "metadata": metadata
            })
            
            # Update context for next iteration
            context_before = chunk_text
        
        # Propagate chapter/section metadata forward
        metadata_list = [c["metadata"] for c in chunks_with_metadata]
        updated_metadata = document_metadata_extractor.propagate_chapter_metadata(metadata_list)
        
        # Update with propagated metadata
        for i, chunk_data in enumerate(chunks_with_metadata):
            chunk_data["metadata"] = updated_metadata[i]
        
        chat_logger.info(f"Created {len(chunks_with_metadata)} chunks with rich metadata")
        
        # Log metadata statistics
        chapters_found = set(m.get("chapter_number") for m in updated_metadata if m.get("chapter_number"))
        sections_found = set(m.get("section_number") for m in updated_metadata if m.get("section_number"))
        
        chat_logger.info(f"Metadata stats: {len(chapters_found)} chapters, {len(sections_found)} sections found")
        
        return chunks_with_metadata

chunking_service = ChunkingService()
