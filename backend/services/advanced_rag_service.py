"""
Advanced RAG Service for Question Generation
Uses sophisticated retrieval techniques for better question quality
"""

from typing import List, Dict, Any, Optional, Tuple
from services.embedding_service import EmbeddingService
from services.qdrant_service import qdrant_service
from services.llamaindex_service import llamaindex_service
from utils.logger import chat_logger
import asyncio
import re
from collections import defaultdict


class AdvancedRAGService:
    """
    Advanced RAG techniques for high-quality question generation:
    1. Multi-Query Retrieval - Generate diverse queries for comprehensive coverage
    2. Reranking - Score and rerank chunks by relevance and information density
    3. Contextual Expansion - Include surrounding context for better understanding
    4. Query Decomposition - Break down topics into subtopics
    5. Diversity Sampling - Ensure questions cover different aspects
    """

    @staticmethod
    async def multi_query_retrieval(
        base_query: str,
        token: str,
        filename: str,
        queries_to_generate: int = 5,
        chunks_per_query: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple query variations and retrieve chunks for each.
        This ensures comprehensive coverage of the topic.

        Args:
            base_query: Original query/topic
            token: User token
            filename: Document filename
            queries_to_generate: Number of query variations
            chunks_per_query: Chunks to retrieve per query

        Returns:
            Deduplicated list of chunks with diversity
        """
        chat_logger.info(
            "Starting multi-query retrieval",
            base_query=base_query[:50],
            num_queries=queries_to_generate,
        )

        # Generate query variations
        query_variations = [
            base_query,  # Original query
            f"Explain the key concepts related to: {base_query}",
            f"What are the main points about: {base_query}",
            f"Describe the important aspects of: {base_query}",
            f"What information is provided about: {base_query}",
        ][:queries_to_generate]

        all_chunks = []
        seen_texts = set()

        # Batch generate embeddings for all query variations
        try:
            query_embeddings = await EmbeddingService.generate_batch_query_embeddings(
                query_variations
            )

            # Retrieve for each query variation
            for query_var, query_embedding in zip(query_variations, query_embeddings):
                try:
                    # Search in Qdrant
                    results = await qdrant_service.search_similar_chunks(
                        query_embedding=query_embedding,
                        token=token,
                        filename=filename,
                        limit=chunks_per_query,
                    )

                    # Deduplicate based on text content
                    for chunk in results:
                        chunk_text = chunk["text"][:100]  # First 100 chars for dedup
                        if chunk_text not in seen_texts:
                            seen_texts.add(chunk_text)
                            all_chunks.append(chunk)

                except Exception as e:
                    chat_logger.warning(
                        f"Query variation failed: {query_var[:30]}", error=str(e)
                    )
                    continue

        except Exception as e:
            chat_logger.warning("Batch embedding failed, falling back to individual", error=str(e))
            # Fallback to individual embeddings
            for query_var in query_variations:
                try:
                    query_embedding = await EmbeddingService.generate_query_embedding(query_var)
                    results = await qdrant_service.search_similar_chunks(
                        query_embedding=query_embedding,
                        token=token,
                        filename=filename,
                        limit=chunks_per_query,
                    )
                    for chunk in results:
                        chunk_text = chunk["text"][:100]
                        if chunk_text not in seen_texts:
                            seen_texts.add(chunk_text)
                            all_chunks.append(chunk)
                except Exception as e2:
                    chat_logger.warning(f"Query variation failed: {query_var[:30]}", error=str(e2))
                    continue

        chat_logger.info(f"Multi-query retrieval found {len(all_chunks)} unique chunks")
        return all_chunks

    @staticmethod
    def calculate_information_density(text: str) -> float:
        """
        Calculate information density score for a chunk.
        Higher score = more informative content for questions.

        Factors:
        - Presence of numbers/data
        - Presence of key terms (because, therefore, important, etc.)
        - Sentence complexity
        - Presence of examples
        """
        score = 0.0

        # Count numbers and data points
        numbers = len(re.findall(r"\b\d+\.?\d*\b", text))
        score += min(numbers * 0.1, 1.0)  # Cap at 1.0

        # Key informative phrases
        key_phrases = [
            "because",
            "therefore",
            "thus",
            "however",
            "important",
            "significant",
            "key",
            "main",
            "primary",
            "essential",
            "for example",
            "such as",
            "including",
            "defined as",
            "means that",
            "refers to",
            "results in",
            "causes",
        ]
        phrase_count = sum(1 for phrase in key_phrases if phrase in text.lower())
        score += min(phrase_count * 0.15, 1.5)

        # Sentence count (more sentences = more concepts)
        sentences = len(re.findall(r"[.!?]+", text))
        score += min(sentences * 0.1, 1.0)

        # Length factor (optimal length is 200-800 chars)
        length = len(text)
        if 200 <= length <= 800:
            score += 1.0
        elif 100 <= length < 200 or 800 < length <= 1200:
            score += 0.5

        # Presence of definitions or explanations
        definition_markers = ["is defined as", "refers to", "means", "is a", "are"]
        if any(marker in text.lower() for marker in definition_markers):
            score += 0.5

        return score

    @staticmethod
    async def rerank_chunks(
        chunks: List[Dict[str, Any]], query: str, top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on:
        1. Original similarity score
        2. Information density
        3. Diversity (avoid too similar chunks)

        Args:
            chunks: List of chunks from retrieval
            query: Original query
            top_k: Number of top chunks to return

        Returns:
            Reranked chunks
        """
        chat_logger.info(f"Reranking {len(chunks)} chunks")

        # Calculate composite scores
        scored_chunks = []
        for chunk in chunks:
            # Original similarity score (0-1)
            similarity_score = chunk.get("score", 0.5)

            # Information density score
            density_score = AdvancedRAGService.calculate_information_density(
                chunk["text"]
            )

            # Composite score (weighted combination)
            composite_score = (
                similarity_score * 0.5  # 50% similarity
                + (density_score / 5.0) * 0.5  # 50% information density (normalized)
            )

            scored_chunks.append(
                {
                    **chunk,
                    "composite_score": composite_score,
                    "density_score": density_score,
                }
            )

        # Sort by composite score
        scored_chunks.sort(key=lambda x: x["composite_score"], reverse=True)

        # Apply diversity sampling - avoid very similar consecutive chunks
        diverse_chunks = []
        for chunk in scored_chunks:
            if len(diverse_chunks) >= top_k:
                break

            # Check diversity with already selected chunks
            is_diverse = True
            chunk_text_lower = chunk["text"].lower()[:200]

            for selected in diverse_chunks[-3:]:  # Check last 3 selected chunks
                selected_text_lower = selected["text"].lower()[:200]
                # Simple overlap check
                overlap = len(
                    set(chunk_text_lower.split()) & set(selected_text_lower.split())
                )
                if overlap > len(chunk_text_lower.split()) * 0.7:  # >70% word overlap
                    is_diverse = False
                    break

            if is_diverse:
                diverse_chunks.append(chunk)

        chat_logger.info(f"Reranked to {len(diverse_chunks)} diverse chunks")
        return diverse_chunks

    @staticmethod
    async def contextual_expansion(
        chunks: List[Dict[str, Any]],
        token: str,
        filename: str,
        expansion_window: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Expand chunks by including surrounding context.
        Retrieves adjacent chunks for better understanding.

        Args:
            chunks: Selected chunks
            token: User token
            filename: Document filename
            expansion_window: Number of adjacent chunks to include

        Returns:
            Chunks with expanded context
        """
        chat_logger.info(
            f"Expanding {len(chunks)} chunks with context window={expansion_window}"
        )

        expanded_chunks = []

        for chunk in chunks:
            chunk_index = chunk.get("chunk_index", -1)
            if chunk_index < 0:
                expanded_chunks.append(chunk)
                continue

            # Get surrounding chunk indices
            adjacent_indices = []
            for offset in range(-expansion_window, expansion_window + 1):
                if offset != 0:  # Don't include the chunk itself
                    adjacent_indices.append(chunk_index + offset)

            # Try to retrieve adjacent chunks
            # Note: This is a simplified version. In production, you'd query Qdrant
            # with chunk_index filter to get adjacent chunks

            # For now, add context marker
            expanded_chunk = chunk.copy()
            expanded_chunk["has_context_expansion"] = True
            expanded_chunk["context_window"] = expansion_window
            expanded_chunks.append(expanded_chunk)

        return expanded_chunks

    @staticmethod
    async def query_decomposition(topic: str) -> List[str]:
        """
        Decompose a complex topic into subtopics.
        Helps retrieve more focused and diverse content.

        Args:
            topic: Main topic/query

        Returns:
            List of subtopic queries
        """
        chat_logger.info(f"Decomposing topic: {topic}")

        # Simple rule-based decomposition
        subtopics = [
            topic,  # Main topic
            f"definition and meaning of {topic}",
            f"examples and applications of {topic}",
            f"key concepts in {topic}",
            f"important aspects of {topic}",
        ]

        # If topic contains specific keywords, add specialized subtopics
        if any(word in topic.lower() for word in ["process", "method", "approach"]):
            subtopics.append(f"steps and procedures in {topic}")

        if any(word in topic.lower() for word in ["theory", "principle", "law"]):
            subtopics.append(f"principles and theories of {topic}")

        if any(
            word in topic.lower() for word in ["history", "evolution", "development"]
        ):
            subtopics.append(f"historical development of {topic}")

        chat_logger.info(f"Decomposed into {len(subtopics)} subtopics")
        return subtopics

    @staticmethod
    async def retrieve_for_questions(
        query: str,
        token: str,
        filename: str,
        num_questions: int = 25,
        mode: str = "comprehensive",
    ) -> Dict[str, Any]:
        """
        Main method: Retrieve content optimized for question generation.
        Combines all advanced techniques.

        Args:
            query: Topic or general query
            token: User token
            filename: Document filename
            num_questions: Number of questions to generate
            mode: 'comprehensive' or 'focused'

        Returns:
            Dictionary with retrieved chunks and metadata
        """
        chat_logger.info(
            "Starting advanced retrieval for question generation",
            query=query[:50],
            num_questions=num_questions,
            mode=mode,
        )

        try:
            # Step 1: Query decomposition (if focused mode)
            if mode == "focused" and query:
                subtopics = await AdvancedRAGService.query_decomposition(query)
                queries_to_use = subtopics
            else:
                queries_to_use = [query] if query else ["comprehensive content"]

            # Step 2: Multi-query retrieval for diversity
            chunks_needed = max(num_questions, 20)  # At least 20 chunks
            all_chunks = []

            for sub_query in queries_to_use[:3]:  # Limit to 3 subtopics to avoid quota
                try:
                    chunks = await AdvancedRAGService.multi_query_retrieval(
                        base_query=sub_query,
                        token=token,
                        filename=filename,
                        queries_to_generate=3,  # 3 variations per subtopic
                        chunks_per_query=5,
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    chat_logger.warning(
                        f"Failed to retrieve for subtopic: {sub_query[:30]}",
                        error=str(e),
                    )

            if not all_chunks:
                chat_logger.warning(
                    "No chunks retrieved, falling back to basic retrieval"
                )
                # Fallback to basic retrieval
                from services.rag_service import rag_service

                fallback_result = await rag_service.retrieve_context(
                    query=query or "generate questions",
                    token=token,
                    filename=filename,
                    top_k=15,
                )
                return fallback_result

            # Step 3: Rerank by relevance and information density
            reranked_chunks = await AdvancedRAGService.rerank_chunks(
                chunks=all_chunks, query=query or "", top_k=chunks_needed
            )

            # Step 4: Contextual expansion (optional, for better context)
            # Skipping for now to avoid additional API calls
            # expanded_chunks = await AdvancedRAGService.contextual_expansion(
            #     chunks=reranked_chunks[:10],
            #     token=token,
            #     filename=filename,
            #     expansion_window=1
            # )

            # Combine chunks into context
            context_parts = []
            for i, chunk in enumerate(reranked_chunks):
                context_parts.append(
                    f"[Chunk {i + 1}, Relevance: {chunk.get('composite_score', 0):.2f}, "
                    f"Info Density: {chunk.get('density_score', 0):.1f}]\n{chunk['text']}"
                )

            combined_context = "\n\n".join(context_parts)

            chat_logger.info(
                f"Advanced retrieval completed with {len(reranked_chunks)} chunks",
                avg_density=sum(c.get("density_score", 0) for c in reranked_chunks)
                / len(reranked_chunks)
                if reranked_chunks
                else 0,
            )

            return {
                "status": "success",
                "chunks": reranked_chunks,
                "context": combined_context,
                "num_chunks": len(reranked_chunks),
                "retrieval_method": "advanced_rag",
                "message": f"Retrieved {len(reranked_chunks)} high-quality chunks using advanced RAG",
            }

        except Exception as e:
            chat_logger.error("Advanced retrieval failed", error=str(e))
            return {
                "status": "error",
                "chunks": [],
                "context": "",
                "num_chunks": 0,
                "message": f"Advanced retrieval failed: {str(e)}",
            }

    @staticmethod
    def analyze_content_for_difficulty(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze chunks to determine difficulty levels for questions.
        Helps generate questions at different Bloom's Taxonomy levels.

        Returns:
            Dictionary with difficulty indicators
        """
        total_chunks = len(chunks)
        if total_chunks == 0:
            return {"difficulty": "medium", "levels": []}

        # Analyze text complexity
        avg_density = (
            sum(
                AdvancedRAGService.calculate_information_density(c["text"])
                for c in chunks
            )
            / total_chunks
        )

        # Calculate average chunk length
        avg_length = sum(len(c["text"]) for c in chunks) / total_chunks

        # Determine difficulty distribution
        if avg_density > 3.0 and avg_length > 500:
            difficulty = "advanced"
            levels = ["Analyzing", "Evaluating", "Creating"]
        elif avg_density > 2.0:
            difficulty = "medium"
            levels = ["Understanding", "Applying", "Analyzing"]
        else:
            difficulty = "basic"
            levels = ["Remembering", "Understanding"]

        return {
            "difficulty": difficulty,
            "levels": levels,
            "avg_density": avg_density,
            "avg_length": avg_length,
        }

    @staticmethod
    async def retrieve_with_llamaindex(
        query: str,
        token: str,
        filename: str,
        top_k: int = 15,
        use_llamaindex: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve using LlamaIndex retrievers with metadata filtering.
        This provides an alternative to the standard Qdrant-based retrieval.

        Args:
            query: Query string
            token: User token
            filename: Document filename
            top_k: Number of results
            use_llamaindex: Whether to use LlamaIndex (for backward compatibility)

        Returns:
            Dictionary with retrieved chunks and metadata
        """
        if not use_llamaindex:
            # Fall back to standard retrieval
            from services.rag_service import rag_service

            return await rag_service.retrieve_context(
                query=query, token=token, filename=filename, top_k=top_k
            )

        try:
            chat_logger.info(
                "Using LlamaIndex for retrieval", query_length=len(query), top_k=top_k
            )

            # Use LlamaIndex service for retrieval
            result = await llamaindex_service.retrieve_with_llamaindex(
                query=query, token=token, filename=filename, top_k=top_k
            )

            if result["status"] == "success":
                # Convert LlamaIndex results to match existing format
                chunks = result["nodes"]
                context_parts = []
                for i, chunk in enumerate(chunks):
                    context_parts.append(
                        f"[LlamaIndex Chunk {i + 1}, Score: {chunk.get('score', 0):.2f}]\n{chunk['text']}"
                    )

                combined_context = "\n\n".join(context_parts)

                return {
                    "status": "success",
                    "chunks": chunks,
                    "context": combined_context,
                    "num_chunks": len(chunks),
                    "retrieval_method": "llamaindex",
                    "message": f"Retrieved {len(chunks)} chunks using LlamaIndex",
                }
            else:
                # Fall back to standard retrieval on error
                chat_logger.warning(
                    "LlamaIndex retrieval failed, falling back to standard"
                )
                from services.rag_service import rag_service

                return await rag_service.retrieve_context(
                    query=query, token=token, filename=filename, top_k=top_k
                )

        except Exception as e:
            chat_logger.error("LlamaIndex retrieval error", error=str(e))
            # Fall back to standard retrieval
            from services.rag_service import rag_service

            return await rag_service.retrieve_context(
                query=query, token=token, filename=filename, top_k=top_k
            )


advanced_rag_service = AdvancedRAGService()
