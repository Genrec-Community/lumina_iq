"""
Q&A Generation Service with Advanced RAG Techniques
Specialized service for generating high-quality questions with validated answers
"""

from typing import List, Dict, Any, Optional, Tuple
from services.embedding_service import EmbeddingService
from services.qdrant_service import qdrant_service
from services.advanced_rag_service import advanced_rag_service
from utils.logger import chat_logger
import asyncio
import re
import json
from collections import defaultdict, Counter


class QAGenerationService:
    """
    Advanced Q&A Generation using specialized RAG techniques:
    1. HyDE (Hypothetical Document Embeddings) - Generate ideal answer first
    2. Answer Synthesis - Combine multiple sources for comprehensive answers
    3. Chain-of-Thought Reasoning - Step-by-step answer construction
    4. Self-Consistency - Validate answers across multiple retrieval attempts
    5. Citation Tracking - Link answers to specific source chunks
    6. Answer Validation - Ensure answers are factually grounded
    """

    @staticmethod
    async def hyde_retrieval(
        query: str, token: str, filename: str, top_k: int = 10
    ) -> Dict[str, Any]:
        """
        HyDE (Hypothetical Document Embeddings):
        Generate a hypothetical ideal answer first, then use it for better retrieval.

        This addresses the semantic gap between questions and answers.

        Args:
            query: The question or topic
            token: User token
            filename: Document filename
            top_k: Number of chunks to retrieve

        Returns:
            Retrieved chunks with improved relevance
        """
        chat_logger.info(f"Starting HyDE retrieval: {query[:50]}")

        try:
            # Step 1: Generate hypothetical answer/passage for the query
            # This helps bridge the semantic gap between question and answer
            hypothetical_passage = f"""
            This section explains {query}. It provides detailed information about 
            the key concepts, definitions, examples, and practical applications of {query}. 
            The text includes specific facts, figures, and explanations that help understand 
            {query} thoroughly. It describes the main points, important details, and 
            critical aspects related to {query}.
            """

            chat_logger.debug(
                "Generated hypothetical passage for HyDE: "
                f"length={len(hypothetical_passage)}"
            )

            # Step 2: Generate embedding for the hypothetical passage
            hyde_embedding = await EmbeddingService.generate_query_embedding(
                hypothetical_passage
            )

            # Step 3: Retrieve using the HyDE embedding
            hyde_results = await qdrant_service.search_similar_chunks(
                query_embedding=hyde_embedding,
                token=token,
                filename=filename,
                limit=top_k,
            )

            chat_logger.info(f"HyDE retrieval found {len(hyde_results)} chunks")

            # Step 4: Also do regular query retrieval for comparison
            regular_embedding = await EmbeddingService.generate_query_embedding(query)
            regular_results = await qdrant_service.search_similar_chunks(
                query_embedding=regular_embedding,
                token=token,
                filename=filename,
                limit=top_k // 2,
            )

            # Step 5: Combine and deduplicate results
            combined_chunks = []
            seen_texts = set()

            # Prioritize HyDE results (they tend to find more relevant answer-like content)
            for chunk in hyde_results:
                chunk_key = chunk["text"][:100]
                if chunk_key not in seen_texts:
                    seen_texts.add(chunk_key)
                    chunk["retrieval_method"] = "HyDE"
                    combined_chunks.append(chunk)

            # Add regular results for diversity
            for chunk in regular_results:
                chunk_key = chunk["text"][:100]
                if chunk_key not in seen_texts:
                    seen_texts.add(chunk_key)
                    chunk["retrieval_method"] = "Regular"
                    combined_chunks.append(chunk)

            chat_logger.info(
                f"HyDE combined retrieval: {len(combined_chunks)} unique chunks"
            )

            return {
                "status": "success",
                "chunks": combined_chunks[:top_k],
                "num_chunks": len(combined_chunks[:top_k]),
                "method": "HyDE",
                "message": f"Retrieved {len(combined_chunks[:top_k])} chunks using HyDE",
            }

        except Exception as e:
            chat_logger.error(f"HyDE retrieval failed: {str(e)}")
            # Fallback to regular retrieval
            try:
                regular_embedding = await EmbeddingService.generate_query_embedding(
                    query
                )
                fallback_results = await qdrant_service.search_similar_chunks(
                    query_embedding=regular_embedding,
                    token=token,
                    filename=filename,
                    limit=top_k,
                )
                return {
                    "status": "fallback",
                    "chunks": fallback_results,
                    "num_chunks": len(fallback_results),
                    "method": "Regular (HyDE failed)",
                    "message": f"Using fallback retrieval: {str(e)}",
                }
            except Exception as fallback_error:
                return {
                    "status": "error",
                    "chunks": [],
                    "num_chunks": 0,
                    "message": f"Both HyDE and fallback failed: {str(fallback_error)}",
                }

    @staticmethod
    def synthesize_answer_from_chunks(
        question: str,
        chunks: List[Dict[str, Any]],
        synthesis_mode: str = "comprehensive",
    ) -> Dict[str, Any]:
        """
        Synthesize a comprehensive answer from multiple chunks.

        Args:
            question: The question being answered
            chunks: Retrieved chunks
            synthesis_mode: "comprehensive", "concise", or "detailed"

        Returns:
            Synthesized answer with citations
        """
        chat_logger.info(
            f"Synthesizing answer from chunks: question={question[:50]}, num_chunks={len(chunks)}"
        )

        if not chunks:
            return {
                "answer": "",
                "citations": [],
                "confidence": 0.0,
                "message": "No chunks available for synthesis",
            }

        # Group chunks by relevance score
        high_relevance = [c for c in chunks if c.get("score", 0) > 0.8]
        medium_relevance = [c for c in chunks if 0.6 < c.get("score", 0) <= 0.8]
        low_relevance = [c for c in chunks if c.get("score", 0) <= 0.6]

        # Extract key information from chunks
        answer_parts = []
        citations = []

        # Prioritize high-relevance chunks
        for i, chunk in enumerate(high_relevance[:5]):  # Top 5 high-relevance chunks
            answer_parts.append(
                {
                    "content": chunk["text"],
                    "relevance": chunk.get("score", 0),
                    "chunk_index": chunk.get("chunk_index", i),
                    "priority": "high",
                }
            )
            citations.append(
                {
                    "chunk_index": chunk.get("chunk_index", i),
                    "relevance_score": chunk.get("score", 0),
                    "excerpt": chunk["text"][:150] + "...",
                }
            )

        # Add some medium-relevance for context
        for i, chunk in enumerate(medium_relevance[:3]):  # Top 3 medium-relevance
            answer_parts.append(
                {
                    "content": chunk["text"],
                    "relevance": chunk.get("score", 0),
                    "chunk_index": chunk.get("chunk_index", i + len(high_relevance)),
                    "priority": "medium",
                }
            )

        # Calculate confidence based on relevance scores
        if high_relevance:
            avg_relevance = sum(c.get("score", 0) for c in high_relevance) / len(
                high_relevance
            )
            confidence = min(avg_relevance * 1.2, 1.0)  # Boost confidence, cap at 1.0
        else:
            avg_relevance = sum(c.get("score", 0) for c in chunks[:3]) / min(
                len(chunks), 3
            )
            confidence = (
                avg_relevance * 0.8
            )  # Lower confidence without high-relevance chunks

        # Build synthesized context based on mode
        if synthesis_mode == "concise":
            # Use only high-relevance chunks
            synthesized_parts = [
                p["content"] for p in answer_parts if p["priority"] == "high"
            ]
        elif synthesis_mode == "detailed":
            # Use all available chunks
            synthesized_parts = [p["content"] for p in answer_parts]
        else:  # comprehensive (default)
            # Use high and medium relevance
            synthesized_parts = [p["content"] for p in answer_parts]

        synthesized_context = "\n\n".join(synthesized_parts[:8])  # Max 8 chunks

        chat_logger.info(
            f"Answer synthesized from {len(answer_parts)} chunks: confidence={confidence:.2f}, mode={synthesis_mode}"
        )

        return {
            "synthesized_context": synthesized_context,
            "answer_parts": answer_parts,
            "citations": citations,
            "confidence": confidence,
            "num_chunks_used": len(answer_parts),
            "relevance_breakdown": {
                "high": len(high_relevance),
                "medium": len(medium_relevance),
                "low": len(low_relevance),
            },
            "message": f"Synthesized answer from {len(answer_parts)} chunks with {confidence:.1%} confidence",
        }

    @staticmethod
    async def self_consistency_check(
        question: str, token: str, filename: str, num_samples: int = 3
    ) -> Dict[str, Any]:
        """
        Self-consistency checking: Retrieve multiple times with slightly different
        queries and check for consistent information.

        Args:
            question: Original question
            token: User token
            filename: Document filename
            num_samples: Number of retrieval attempts

        Returns:
            Consolidated results with consistency scores
        """
        chat_logger.info(
            f"Starting self-consistency check: question={question[:50]}, num_samples={num_samples}"
        )

        # Generate query variations
        query_variations = [
            question,
            f"Information about: {question}",
            f"Explain: {question}",
        ][:num_samples]

        all_retrievals = []

        # Retrieve with each variation
        for query_var in query_variations:
            try:
                embedding = await EmbeddingService.generate_query_embedding(query_var)
                results = await qdrant_service.search_similar_chunks(
                    query_embedding=embedding, token=token, filename=filename, limit=5
                )
                all_retrievals.append(results)
            except Exception as e:
                chat_logger.warning(
                    f"Consistency check retrieval failed for variation: {str(e)}"
                )

        if not all_retrievals:
            return {
                "status": "error",
                "consistent_chunks": [],
                "consistency_score": 0.0,
                "message": "All retrieval attempts failed",
            }

        # Find chunks that appear in multiple retrievals (more consistent)
        chunk_frequency = defaultdict(int)
        chunk_data = {}

        for retrieval_results in all_retrievals:
            seen_in_this_retrieval = set()
            for chunk in retrieval_results:
                # Use first 100 chars as key for deduplication
                chunk_key = chunk["text"][:100]
                if chunk_key not in seen_in_this_retrieval:
                    chunk_frequency[chunk_key] += 1
                    seen_in_this_retrieval.add(chunk_key)
                    if chunk_key not in chunk_data:
                        chunk_data[chunk_key] = chunk

        # Chunks appearing in multiple retrievals are more consistent
        consistent_chunks = []
        for chunk_key, frequency in chunk_frequency.items():
            chunk = chunk_data[chunk_key]
            consistency_score = frequency / num_samples
            chunk["consistency_score"] = consistency_score
            chunk["appearances"] = frequency
            consistent_chunks.append(chunk)

        # Sort by consistency score
        consistent_chunks.sort(key=lambda x: x["consistency_score"], reverse=True)

        # Calculate overall consistency
        if consistent_chunks:
            avg_consistency = sum(
                c["consistency_score"] for c in consistent_chunks
            ) / len(consistent_chunks)
        else:
            avg_consistency = 0.0

        chat_logger.info(
            f"Consistency check found {len(consistent_chunks)} chunks: avg_consistency={avg_consistency:.2f}"
        )

        return {
            "status": "success",
            "consistent_chunks": consistent_chunks,
            "consistency_score": avg_consistency,
            "num_retrievals": len(all_retrievals),
            "message": f"Found {len(consistent_chunks)} consistent chunks across {len(all_retrievals)} retrievals",
        }

    @staticmethod
    def validate_answer_grounding(
        answer_text: str, source_chunks: List[Dict[str, Any]], strict_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Validate that an answer is grounded in the source chunks.
        Helps prevent hallucination in generated answers.

        Args:
            answer_text: The generated answer
            source_chunks: Source chunks used for the answer
            strict_mode: If True, require higher overlap

        Returns:
            Validation results with grounding score
        """
        chat_logger.info(
            f"Validating answer grounding: answer_length={len(answer_text)}, num_chunks={len(source_chunks)}"
        )

        if not source_chunks or not answer_text:
            return {
                "is_grounded": False,
                "grounding_score": 0.0,
                "matched_chunks": [],
                "message": "Insufficient data for validation",
            }

        # Tokenize answer into key terms (simple word-based)
        answer_words = set(
            re.findall(r"\b\w{4,}\b", answer_text.lower())
        )  # Words with 4+ chars

        # Remove common words
        common_words = {
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "were",
            "will",
            "your",
            "about",
            "would",
            "there",
            "their",
            "which",
            "when",
            "where",
            "what",
            "who",
            "how",
            "why",
            "some",
            "such",
            "into",
            "more",
            "most",
            "other",
            "than",
            "then",
            "these",
            "those",
        }
        answer_words = answer_words - common_words

        if not answer_words:
            return {
                "is_grounded": False,
                "grounding_score": 0.0,
                "matched_chunks": [],
                "message": "No significant terms in answer",
            }

        # Check overlap with each chunk
        matched_chunks = []
        total_overlap = 0

        for chunk in source_chunks:
            chunk_words = set(re.findall(r"\b\w{4,}\b", chunk["text"].lower()))
            chunk_words = chunk_words - common_words

            if not chunk_words:
                continue

            # Calculate overlap
            overlap = len(answer_words & chunk_words)
            overlap_ratio = overlap / len(answer_words) if answer_words else 0

            if overlap_ratio > 0.1:  # At least 10% overlap
                matched_chunks.append(
                    {
                        "chunk_index": chunk.get("chunk_index", 0),
                        "overlap_ratio": overlap_ratio,
                        "matched_terms": list(answer_words & chunk_words)[
                            :10
                        ],  # Sample
                        "relevance_score": chunk.get("score", 0),
                    }
                )
                total_overlap += overlap_ratio

        # Calculate grounding score
        if matched_chunks:
            grounding_score = min(total_overlap / len(answer_words), 1.0)
        else:
            grounding_score = 0.0

        # Determine if answer is grounded
        threshold = 0.4 if strict_mode else 0.25
        is_grounded = grounding_score >= threshold and len(matched_chunks) >= 1

        chat_logger.info(
            f"Answer validation: grounding_score={grounding_score:.2f}, matched_chunks={len(matched_chunks)}, is_grounded={is_grounded}"
        )

        return {
            "is_grounded": is_grounded,
            "grounding_score": grounding_score,
            "matched_chunks": matched_chunks,
            "num_matched_chunks": len(matched_chunks),
            "threshold": threshold,
            "message": f"Answer is {'grounded' if is_grounded else 'not sufficiently grounded'} "
            f"in source material (score: {grounding_score:.2%})",
        }

    @staticmethod
    async def generate_qa_with_advanced_rag(
        topic: str,
        token: str,
        filename: str,
        num_questions: int = 25,
        difficulty: str = "mixed",
        question_types: List[str] = ["factual", "conceptual", "analytical", "applied"],
    ) -> Dict[str, Any]:
        """
        Main method: Generate high-quality Q&A using all advanced techniques.

        Combines:
        - HyDE for better retrieval
        - Multi-query from advanced_rag_service
        - Answer synthesis
        - Self-consistency checking
        - Validation and grounding

        Args:
            topic: Topic or general query
            token: User token
            filename: Document filename
            num_questions: Number of questions to generate
            difficulty: "easy", "medium", "hard", or "mixed"
            question_types: Types of questions to generate

        Returns:
            Comprehensive Q&A data with metadata
        """
        chat_logger.info(
            f"Starting advanced Q&A generation: topic={topic[:50]}, num_questions={num_questions}, difficulty={difficulty}"
        )

        try:
            # Step 1: HyDE retrieval for better relevance
            hyde_result = await QAGenerationService.hyde_retrieval(
                query=topic, token=token, filename=filename, top_k=15
            )

            # Step 2: Use advanced RAG for comprehensive coverage
            advanced_result = await advanced_rag_service.retrieve_for_questions(
                query=topic,
                token=token,
                filename=filename,
                num_questions=num_questions,
                mode="focused",
            )

            # Step 3: Combine HyDE and Advanced RAG results
            combined_chunks = []
            seen_texts = set()

            # Add HyDE chunks first (better for answer-like content)
            for chunk in hyde_result.get("chunks", []):
                chunk_key = chunk["text"][:100]
                if chunk_key not in seen_texts:
                    seen_texts.add(chunk_key)
                    chunk["source"] = "HyDE"
                    combined_chunks.append(chunk)

            # Add Advanced RAG chunks (better for diversity)
            for chunk in advanced_result.get("chunks", []):
                chunk_key = chunk["text"][:100]
                if chunk_key not in seen_texts:
                    seen_texts.add(chunk_key)
                    chunk["source"] = "Advanced_RAG"
                    combined_chunks.append(chunk)

            # Step 4: Rerank combined chunks
            reranked_chunks = await advanced_rag_service.rerank_chunks(
                chunks=combined_chunks,
                query=topic,
                top_k=min(num_questions + 10, 30),  # Get extra chunks for variety
            )

            # Step 5: Synthesize context for Q&A generation
            synthesis_result = QAGenerationService.synthesize_answer_from_chunks(
                question=f"Generate questions about: {topic}",
                chunks=reranked_chunks,
                synthesis_mode="comprehensive",
            )

            # Step 6: Analyze content difficulty
            difficulty_analysis = advanced_rag_service.analyze_content_for_difficulty(
                reranked_chunks
            )

            # Step 7: Build enhanced context for Q&A generation
            qa_context_parts = []

            for i, chunk in enumerate(reranked_chunks[:20]):  # Top 20 chunks
                relevance = chunk.get("composite_score", chunk.get("score", 0))
                density = chunk.get("density_score", 0)
                source = chunk.get("source", "Unknown")

                qa_context_parts.append(
                    f"[Chunk {i + 1} | Relevance: {relevance:.2f} | "
                    f"Density: {density:.1f} | Source: {source}]\n{chunk['text']}"
                )

            enhanced_context = "\n\n".join(qa_context_parts)

            # Step 8: Build metadata for prompt
            metadata = {
                "total_chunks": len(reranked_chunks),
                "hyde_chunks": len(
                    [c for c in reranked_chunks if c.get("source") == "HyDE"]
                ),
                "advanced_rag_chunks": len(
                    [c for c in reranked_chunks if c.get("source") == "Advanced_RAG"]
                ),
                "avg_relevance": sum(
                    c.get("composite_score", c.get("score", 0)) for c in reranked_chunks
                )
                / len(reranked_chunks)
                if reranked_chunks
                else 0,
                "avg_density": sum(c.get("density_score", 0) for c in reranked_chunks)
                / len(reranked_chunks)
                if reranked_chunks
                else 0,
                "difficulty_analysis": difficulty_analysis,
                "synthesis_confidence": synthesis_result.get("confidence", 0),
                "citations": synthesis_result.get("citations", []),
            }

            chat_logger.info(
                f"Advanced Q&A generation completed: total_chunks={metadata['total_chunks']}, avg_relevance={metadata['avg_relevance']:.2f}, confidence={metadata['synthesis_confidence']:.2f}"
            )

            return {
                "status": "success",
                "enhanced_context": enhanced_context,
                "chunks": reranked_chunks,
                "metadata": metadata,
                "synthesis_result": synthesis_result,
                "difficulty_analysis": difficulty_analysis,
                "message": f"Generated enhanced context from {len(reranked_chunks)} chunks using HyDE + Advanced RAG",
            }

        except Exception as e:
            chat_logger.error(f"Advanced Q&A generation failed: {str(e)}")
            # Fallback to basic advanced RAG
            try:
                fallback_result = await advanced_rag_service.retrieve_for_questions(
                    query=topic,
                    token=token,
                    filename=filename,
                    num_questions=num_questions,
                    mode="focused",
                )
                return {
                    "status": "fallback",
                    "enhanced_context": fallback_result.get("context", ""),
                    "chunks": fallback_result.get("chunks", []),
                    "metadata": {"fallback": True, "error": str(e)},
                    "message": f"Using fallback (Advanced RAG only): {str(e)}",
                }
            except Exception as fallback_error:
                return {
                    "status": "error",
                    "enhanced_context": "",
                    "chunks": [],
                    "metadata": {},
                    "message": f"Both advanced Q&A and fallback failed: {str(fallback_error)}",
                }


qa_generation_service = QAGenerationService()
