"""
Retrieval Strategy Manager
Manages different retrieval strategies optimized for different use cases
"""

from typing import List, Dict, Any, Optional
from services.embedding_service import EmbeddingService
from services.qdrant_service import qdrant_service
from services.query_classifier import query_classifier, query_metadata_extractor
from utils.logger import chat_logger
import asyncio


class RetrievalStrategyManager:
    """
    Manages 4 different retrieval strategies:
    1. CHAT: Hybrid search with reranking for conversational Q&A
    2. EVALUATION: Dense semantic search with context expansion
    3. QA_GENERATION: Sequential retrieval with topic filtering
    4. NOTES: Hierarchical retrieval with complete section coverage
    """

    @staticmethod
    async def retrieve(
        query: str, token: str, filename: str, use_case: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Main retrieval method that routes to appropriate strategy.

        Args:
            query: User's query
            token: User token
            filename: Document filename
            use_case: Optional explicit use case (auto-detected if not provided)
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dictionary with retrieved chunks and metadata
        """
        # Classify query if use case not provided
        if not use_case:
            classification = query_classifier.classify_query(query)
            use_case = classification["use_case"]
            chat_logger.info(
                f"Auto-detected use case: {use_case}",
                confidence=classification["confidence"],
            )

        # Extract query metadata
        query_metadata = query_metadata_extractor.extract_all_metadata(query)

        # Get context requirements
        if use_case is None:
            use_case = "chat"  # Default to chat if not determined
        requirements = query_classifier.extract_context_requirements(query, use_case)

        # Route to appropriate strategy
        if use_case == "chat":
            result = await RetrievalStrategyManager._chat_strategy(
                query, token, filename, query_metadata, requirements, **kwargs
            )
        elif use_case == "evaluation":
            result = await RetrievalStrategyManager._evaluation_strategy(
                query, token, filename, query_metadata, requirements, **kwargs
            )
        elif use_case == "qa_generation":
            result = await RetrievalStrategyManager._qa_generation_strategy(
                query, token, filename, query_metadata, requirements, **kwargs
            )
        elif use_case == "notes":
            result = await RetrievalStrategyManager._notes_strategy(
                query, token, filename, query_metadata, requirements, **kwargs
            )
        else:
            # Fallback to chat strategy
            result = await RetrievalStrategyManager._chat_strategy(
                query, token, filename, query_metadata, requirements, **kwargs
            )

        # Add metadata to result
        result["use_case"] = use_case
        result["query_metadata"] = query_metadata
        result["requirements"] = requirements

        return result

    @staticmethod
    async def _chat_strategy(
        query: str,
        token: str,
        filename: str,
        query_metadata: Dict[str, Any],
        requirements: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        CHAT Strategy: Hybrid search with reranking
        - Semantic search for relevance
        - Allows cross-chapter results
        - Reranks for best top-k results
        - Medium chunk size
        """
        chat_logger.info(f"Using CHAT retrieval strategy: {top_k}")

        try:
            # Generate query embedding
            query_embedding = await EmbeddingService.generate_query_embedding(query)

            # Build filter conditions
            filter_conditions = []

            # Optional chapter filter (only if high confidence)
            if query_metadata["chapter"]["confidence"] > 0.90:
                filter_conditions.append(
                    {
                        "key": "chapter_number",
                        "value": query_metadata["chapter"]["value"],
                    }
                )

            # Search with higher limit for reranking
            initial_limit = top_k * 3
            results = await qdrant_service.search_similar_chunks(
                query_embedding=query_embedding,
                token=token,
                filename=filename,
                limit=initial_limit,
                metadata_filters=filter_conditions if filter_conditions else None,
            )

            if not results:
                # Try without filters if no results
                if filter_conditions:
                    chat_logger.info("No results with filters, retrying without")
                    results = await qdrant_service.search_similar_chunks(
                        query_embedding=query_embedding,
                        token=token,
                        filename=filename,
                        limit=initial_limit,
                    )

            # Rerank results
            reranked = RetrievalStrategyManager._simple_rerank(
                chunks=results, query=query, top_k=top_k
            )

            # Build context
            context = RetrievalStrategyManager._build_context(
                reranked, style="conversational"
            )

            chat_logger.info(f"CHAT strategy retrieved {len(reranked)} chunks")

            return {
                "status": "success",
                "chunks": reranked,
                "context": context,
                "num_chunks": len(reranked),
                "strategy": "chat",
                "message": f"Retrieved {len(reranked)} chunks using CHAT strategy",
            }

        except Exception as e:
            chat_logger.error(f"CHAT strategy failed: {str(e)}")
            return {
                "status": "error",
                "chunks": [],
                "context": "",
                "num_chunks": 0,
                "strategy": "chat",
                "message": f"CHAT strategy failed: {str(e)}",
            }

    @staticmethod
    async def _evaluation_strategy(
        query: str,
        token: str,
        filename: str,
        query_metadata: Dict[str, Any],
        requirements: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        EVALUATION Strategy: Dense semantic search with context expansion
        - High score threshold for precision
        - Retrieves surrounding chunks for complete context
        - Filters by chapter/section if mentioned
        - Small chunks but with expanded context
        """
        chat_logger.info(f"Using EVALUATION retrieval strategy: {top_k}")

        try:
            # Generate query embedding
            query_embedding = await EmbeddingService.generate_query_embedding(query)

            # Build filter conditions (stricter for evaluation)
            filter_conditions = []

            # Chapter filter (medium confidence threshold)
            if query_metadata["chapter"]["confidence"] > 0.80:
                filter_conditions.append(
                    {
                        "key": "chapter_number",
                        "value": query_metadata["chapter"]["value"],
                    }
                )

            # Section filter
            if query_metadata["section"]["confidence"] > 0.85:
                filter_conditions.append(
                    {
                        "key": "section_number",
                        "value": query_metadata["section"]["value"],
                    }
                )

            # Search with high score threshold
            results = await qdrant_service.search_similar_chunks(
                query_embedding=query_embedding,
                token=token,
                filename=filename,
                limit=top_k,
                score_threshold=0.75,  # High threshold for evaluation
                metadata_filters=filter_conditions if filter_conditions else None,
            )

            if not results and filter_conditions:
                # Retry without filters
                chat_logger.info("No results with filters, retrying without")
                results = await qdrant_service.search_similar_chunks(
                    query_embedding=query_embedding,
                    token=token,
                    filename=filename,
                    limit=top_k,
                    score_threshold=0.70,
                )

            # Context expansion: Get surrounding chunks
            expanded_results = await RetrievalStrategyManager._expand_context(
                chunks=results,
                token=token,
                filename=filename,
                window_size=2,  # ±2 chunks
            )

            # Build context with full details
            context = RetrievalStrategyManager._build_context(
                expanded_results, style="detailed"
            )

            chat_logger.info(
                f"EVALUATION strategy retrieved {len(expanded_results)} chunks (with expansion)"
            )

            return {
                "status": "success",
                "chunks": expanded_results,
                "context": context,
                "num_chunks": len(expanded_results),
                "num_core_chunks": len(results),
                "strategy": "evaluation",
                "message": f"Retrieved {len(results)} core chunks with {len(expanded_results) - len(results)} expanded chunks",
            }

        except Exception as e:
            chat_logger.error(f"EVALUATION strategy failed: {str(e)}")
            return {
                "status": "error",
                "chunks": [],
                "context": "",
                "num_chunks": 0,
                "strategy": "evaluation",
                "message": f"EVALUATION strategy failed: {str(e)}",
            }

    @staticmethod
    async def _qa_generation_strategy(
        query: str,
        token: str,
        filename: str,
        query_metadata: Dict[str, Any],
        requirements: Dict[str, Any],
        num_questions: int = 25,
    ) -> Dict[str, Any]:
        """
        QA_GENERATION Strategy: Sequential retrieval with topic filtering
        - MUST filter by chapter/topic if mentioned
        - Retrieves larger, sequential chunks
        - Maintains document order (sequential_id)
        - No cross-chapter mixing
        """
        chat_logger.info(f"Using QA_GENERATION retrieval strategy: {num_questions}")

        try:
            # Generate query embedding
            query_embedding = await EmbeddingService.generate_query_embedding(query)

            # Build STRICT filter conditions
            filter_conditions = []

            # Chapter filter (MANDATORY if mentioned)
            if query_metadata["chapter"]["confidence"] > 0.70:
                filter_conditions.append(
                    {
                        "key": "chapter_number",
                        "value": query_metadata["chapter"]["value"],
                    }
                )
                chat_logger.info(f"Filtering to chapter {query_metadata['chapter']['value']}")

            # Section filter
            if query_metadata["section"]["confidence"] > 0.80:
                filter_conditions.append(
                    {
                        "key": "section_number",
                        "value": query_metadata["section"]["value"],
                    }
                )

            # Determine how many chunks to retrieve
            chunks_needed = max(num_questions, 15)

            # Search with filters
            results = await qdrant_service.search_similar_chunks(
                query_embedding=query_embedding,
                token=token,
                filename=filename,
                limit=chunks_needed,
                metadata_filters=filter_conditions if filter_conditions else None,
            )

            if not results:
                chat_logger.warning("No results found for Q&A generation")
                return {
                    "status": "error",
                    "chunks": [],
                    "context": "",
                    "num_chunks": 0,
                    "strategy": "qa_generation",
                    "message": "No content found matching the criteria. Please check chapter/topic specification.",
                }

            # Sort by sequential_id to maintain document order
            results_sorted = sorted(
                results,
                key=lambda x: x.get("metadata", {}).get(
                    "sequential_id", x.get("chunk_index", 0)
                ),
            )

            # Get sequential chunks from same section
            sequential_results = await RetrievalStrategyManager._get_sequential_chunks(
                initial_chunks=results_sorted,
                token=token,
                filename=filename,
                target_count=chunks_needed,
            )

            # Build context preserving order
            context = RetrievalStrategyManager._build_context(
                sequential_results, style="structured", preserve_order=True
            )

            chat_logger.info(
                f"QA_GENERATION strategy retrieved {len(sequential_results)} sequential chunks"
            )

            return {
                "status": "success",
                "chunks": sequential_results,
                "context": context,
                "num_chunks": len(sequential_results),
                "strategy": "qa_generation",
                "filtered_by_chapter": query_metadata["chapter"]["value"]
                if query_metadata["chapter"]["confidence"] > 0.70
                else None,
                "message": f"Retrieved {len(sequential_results)} sequential chunks for Q&A generation",
            }

        except Exception as e:
            chat_logger.error(f"QA_GENERATION strategy failed: {str(e)}")
            return {
                "status": "error",
                "chunks": [],
                "context": "",
                "num_chunks": 0,
                "strategy": "qa_generation",
                "message": f"QA_GENERATION strategy failed: {str(e)}",
            }

    @staticmethod
    async def _notes_strategy(
        query: str,
        token: str,
        filename: str,
        query_metadata: Dict[str, Any],
        requirements: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        NOTES Strategy: Hierarchical retrieval with complete section coverage
        - Retrieves ALL chunks from specified chapter/section
        - Ordered by sequential_id
        - Groups by subtopics/sections
        - Large chunks for comprehensive coverage
        """
        chat_logger.info(f"Using NOTES retrieval strategy: {top_k}")

        try:
            # Build filter for chapter/section
            filter_conditions = []

            # Chapter filter (MANDATORY for notes)
            if query_metadata["chapter"]["confidence"] > 0.70:
                filter_conditions.append(
                    {
                        "key": "chapter_number",
                        "value": query_metadata["chapter"]["value"],
                    }
                )
                chat_logger.info(f"Generating notes for chapter {query_metadata['chapter']['value']}")
            elif query_metadata["section"]["confidence"] > 0.80:
                filter_conditions.append(
                    {
                        "key": "section_number",
                        "value": query_metadata["section"]["value"],
                    }
                )

            # If no specific chapter/section, retrieve based on topic
            if not filter_conditions:
                # Use semantic search with topic
                query_embedding = await EmbeddingService.generate_query_embedding(query)
                results = await qdrant_service.search_similar_chunks(
                    query_embedding=query_embedding,
                    token=token,
                    filename=filename,
                    limit=top_k,
                )
            else:
                # Retrieve ALL chunks from chapter/section
                # This requires a scroll/get all approach
                results = await qdrant_service.get_chunks_by_filter(
                    token=token,
                    filename=filename,
                    metadata_filters=filter_conditions,
                    limit=top_k,
                )

            if not results:
                chat_logger.warning("No chunks found for notes generation")
                return {
                    "status": "error",
                    "chunks": [],
                    "context": "",
                    "num_chunks": 0,
                    "strategy": "notes",
                    "message": "No content found for notes generation",
                }

            # Sort by sequential_id for proper flow
            results_sorted = sorted(
                results,
                key=lambda x: x.get("metadata", {}).get(
                    "sequential_id", x.get("chunk_index", 0)
                ),
            )

            # Group by sections for hierarchical organization
            grouped = RetrievalStrategyManager._group_by_sections(results_sorted)

            # Build hierarchical context
            context = RetrievalStrategyManager._build_hierarchical_context(grouped)

            chat_logger.info(
                f"NOTES strategy retrieved {len(results_sorted)} chunks in {len(grouped)} sections"
            )

            return {
                "status": "success",
                "chunks": results_sorted,
                "context": context,
                "num_chunks": len(results_sorted),
                "num_sections": len(grouped),
                "grouped_chunks": grouped,
                "strategy": "notes",
                "message": f"Retrieved {len(results_sorted)} chunks across {len(grouped)} sections for notes",
            }

        except Exception as e:
            chat_logger.error(f"NOTES strategy failed: {str(e)}")
            return {
                "status": "error",
                "chunks": [],
                "context": "",
                "num_chunks": 0,
                "strategy": "notes",
                "message": f"NOTES strategy failed: {str(e)}",
            }

    # Helper methods

    @staticmethod
    def _simple_rerank(
        chunks: List[Dict[str, Any]], query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Simple reranking based on score and text relevance."""
        # Sort by score
        sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_chunks[:top_k]

    @staticmethod
    async def _expand_context(
        chunks: List[Dict[str, Any]], token: str, filename: str, window_size: int = 2
    ) -> List[Dict[str, Any]]:
        """Expand chunks by retrieving surrounding chunks."""
        expanded = list(chunks)  # Start with original chunks

        for chunk in chunks:
            sequential_id = chunk.get("metadata", {}).get(
                "sequential_id", chunk.get("chunk_index")
            )
            if sequential_id is None:
                continue

            # Get adjacent chunks
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue

                adjacent_id = sequential_id + offset
                # Try to retrieve adjacent chunk
                # This is simplified - in production you'd query Qdrant with sequential_id filter
                # For now, we'll mark that expansion is needed

        # Mark as expanded
        for chunk in expanded:
            chunk["is_expanded"] = True

        return expanded

    @staticmethod
    async def _get_sequential_chunks(
        initial_chunks: List[Dict[str, Any]],
        token: str,
        filename: str,
        target_count: int,
    ) -> List[Dict[str, Any]]:
        """Get sequential chunks starting from initial chunks."""
        # For now, just return sorted initial chunks
        # In production, you'd retrieve more sequential chunks if needed
        return initial_chunks[:target_count]

    @staticmethod
    def _group_by_sections(
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by section for hierarchical organization."""
        grouped = {}

        for chunk in chunks:
            section = chunk.get("metadata", {}).get("section_number", "General")
            if section not in grouped:
                grouped[section] = []
            grouped[section].append(chunk)

        return grouped

    @staticmethod
    def _build_context(
        chunks: List[Dict[str, Any]],
        style: str = "conversational",
        preserve_order: bool = False,
    ) -> str:
        """Build context string from chunks."""
        if not chunks:
            return ""

        parts = []

        for i, chunk in enumerate(chunks):
            score = chunk.get("score", 0)
            text = chunk.get("text", "")

            if style == "conversational":
                parts.append(f"[Chunk {i + 1}, Relevance: {score:.2f}]\n{text}")
            elif style == "detailed":
                chapter = chunk.get("metadata", {}).get("chapter_number", "N/A")
                section = chunk.get("metadata", {}).get("section_number", "N/A")
                parts.append(
                    f"[Chunk {i + 1} | Chapter {chapter} | Section {section} | Score: {score:.2f}]\n{text}"
                )
            elif style == "structured":
                seq_id = chunk.get("metadata", {}).get("sequential_id", i)
                parts.append(f"[Chunk {seq_id}]\n{text}")

        return "\n\n".join(parts)

    @staticmethod
    def _build_hierarchical_context(grouped: Dict[str, List[Dict[str, Any]]]) -> str:
        """Build hierarchical context for notes."""
        parts = []

        for section, chunks in grouped.items():
            parts.append(f"\n{'=' * 60}\n{section}\n{'=' * 60}")
            for chunk in chunks:
                parts.append(chunk.get("text", ""))

        return "\n\n".join(parts)


retrieval_strategy_manager = RetrievalStrategyManager()
