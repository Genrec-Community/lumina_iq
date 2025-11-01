"""
Embedding Service for Lumina IQ RAG Backend.

Handles embedding generation using LangChain with Together AI as the provider.
"""

from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from utils.logger import get_logger
from .cache_service import cache_service

logger = get_logger("embedding_service")


class EmbeddingService:
    """Service for generating embeddings using LangChain + Together AI."""

    def __init__(self):
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.is_initialized = False

    def initialize(self) -> None:
        """Initialize LangChain embeddings with Together AI."""
        try:
            if not settings.TOGETHER_API_KEY:
                logger.warning("TOGETHER_API_KEY not configured")
                return

            logger.info(
                "Initializing embedding service",
                extra={
                    "extra_fields": {
                        "model": settings.EMBEDDING_MODEL,
                        "dimensions": settings.EMBEDDING_DIMENSIONS,
                        "provider": "Together AI (via LangChain)",
                    }
                },
            )

            # Initialize LangChain OpenAIEmbeddings with Together AI base URL
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_base=settings.TOGETHER_BASE_URL,
                openai_api_key=settings.TOGETHER_API_KEY,
                dimensions=settings.EMBEDDING_DIMENSIONS,
            )

            self.is_initialized = True
            logger.info("Embedding service initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize embedding service: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            self.is_initialized = False
            raise

    async def generate_embedding(
        self, text: str, use_cache: bool = True
    ) -> List[float]:
        """Generate embedding for text with optional caching."""
        if not self.is_initialized or not self.embeddings:
            raise RuntimeError("Embedding service not initialized")

        try:
            # Check cache first if enabled
            if use_cache and settings.CACHE_EMBEDDINGS:
                cached_embedding = await cache_service.get_cached_embedding(
                    text, settings.EMBEDDING_MODEL
                )
                if cached_embedding:
                    logger.debug(
                        "Using cached embedding",
                        extra={"extra_fields": {"text_length": len(text)}},
                    )
                    return cached_embedding

            logger.debug(
                f"Generating embedding for text",
                extra={
                    "extra_fields": {
                        "text_length": len(text),
                        "model": settings.EMBEDDING_MODEL,
                    }
                },
            )

            # Generate embedding using LangChain
            embedding = await self.embeddings.aembed_query(text)

            # Cache the embedding if enabled
            if use_cache and settings.CACHE_EMBEDDINGS:
                await cache_service.cache_embedding(
                    text, embedding, settings.EMBEDDING_MODEL
                )

            logger.debug(
                "Generated embedding successfully",
                extra={
                    "extra_fields": {
                        "embedding_dim": len(embedding),
                        "text_length": len(text),
                    }
                },
            )

            return embedding

        except Exception as e:
            logger.error(
                f"Failed to generate embedding: {str(e)}",
                extra={
                    "extra_fields": {
                        "error_type": type(e).__name__,
                        "text_length": len(text),
                    }
                },
            )
            raise

    async def generate_embeddings_batch(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching and caching."""
        if not self.is_initialized or not self.embeddings:
            raise RuntimeError("Embedding service not initialized")

        try:
            logger.info(
                f"Generating embeddings for batch",
                extra={
                    "extra_fields": {
                        "batch_size": len(texts),
                        "model": settings.EMBEDDING_MODEL,
                    }
                },
            )

            embeddings = []
            texts_to_embed = []
            cache_indices = []

            # Check cache for each text if enabled
            if use_cache and settings.CACHE_EMBEDDINGS:
                for i, text in enumerate(texts):
                    cached_embedding = await cache_service.get_cached_embedding(
                        text, settings.EMBEDDING_MODEL
                    )
                    if cached_embedding:
                        embeddings.append((i, cached_embedding))
                    else:
                        texts_to_embed.append(text)
                        cache_indices.append(i)

                logger.debug(
                    f"Cache stats for batch",
                    extra={
                        "extra_fields": {
                            "cached": len(embeddings),
                            "to_generate": len(texts_to_embed),
                        }
                    },
                )
            else:
                texts_to_embed = texts
                cache_indices = list(range(len(texts)))

            # Generate embeddings for uncached texts
            if texts_to_embed:
                # Process in batches to avoid rate limits
                batch_size = settings.EMBEDDING_BATCH_SIZE
                new_embeddings = []

                for i in range(0, len(texts_to_embed), batch_size):
                    batch = texts_to_embed[i : i + batch_size]
                    batch_embeddings = await self.embeddings.aembed_documents(batch)
                    new_embeddings.extend(batch_embeddings)

                    logger.debug(
                        f"Generated batch embeddings",
                        extra={
                            "extra_fields": {
                                "batch_start": i,
                                "batch_size": len(batch),
                            }
                        },
                    )

                # Cache new embeddings if enabled
                if use_cache and settings.CACHE_EMBEDDINGS:
                    for text, embedding in zip(texts_to_embed, new_embeddings):
                        await cache_service.cache_embedding(
                            text, embedding, settings.EMBEDDING_MODEL
                        )

                # Merge with cached embeddings
                for idx, embedding in zip(cache_indices, new_embeddings):
                    embeddings.append((idx, embedding))

            # Sort by original index to maintain order
            embeddings.sort(key=lambda x: x[0])
            result_embeddings = [emb for _, emb in embeddings]

            logger.info(
                "Generated batch embeddings successfully",
                extra={
                    "extra_fields": {
                        "total_embeddings": len(result_embeddings),
                        "embedding_dim": (
                            len(result_embeddings[0]) if result_embeddings else 0
                        ),
                    }
                },
            )

            return result_embeddings

        except Exception as e:
            logger.error(
                f"Failed to generate batch embeddings: {str(e)}",
                extra={
                    "extra_fields": {
                        "error_type": type(e).__name__,
                        "batch_size": len(texts),
                    }
                },
            )
            raise

    async def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            import math

            # Compute dot product
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

            # Compute magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in embedding1))
            magnitude2 = math.sqrt(sum(b * b for b in embedding2))

            # Compute cosine similarity
            similarity = dot_product / (magnitude1 * magnitude2)

            return similarity

        except Exception as e:
            logger.error(
                f"Failed to compute similarity: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def find_similar_texts(
        self, query_embedding: List[float], candidate_embeddings: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """Find most similar texts to query embedding."""
        try:
            similarities = []

            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = await self.compute_similarity(
                    query_embedding, candidate_embedding
                )
                similarities.append({"index": i, "similarity": similarity})

            # Sort by similarity descending
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            return similarities

        except Exception as e:
            logger.error(
                f"Failed to find similar texts: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise


# Global singleton instance
embedding_service = EmbeddingService()
