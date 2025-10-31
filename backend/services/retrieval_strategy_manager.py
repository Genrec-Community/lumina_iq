"""
Retrieval strategy manager for RAG pipeline.
Manages different retrieval strategies and selects the best one.
"""

from typing import List, Dict, Any, Optional
from services.qdrant_service import qdrant_service
from services.embedding_service import embedding_service
from utils.logger import get_logger

logger = get_logger("retrieval_strategy_manager")


class RetrievalStrategyManager:
    """Manages different retrieval strategies for RAG"""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Initialize the retrieval strategy manager"""
        if not self._initialized:
            qdrant_service.initialize()
            self._initialized = True

    async def retrieve_dense(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense retrieval using semantic embeddings.

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            List of retrieved documents
        """
        if not self._initialized:
            self.initialize()

        try:
            results = await qdrant_service.search(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
            )

            logger.info(
                f"Dense retrieval completed: {len(results)} results",
                extra={"extra_fields": {"query": query[:100], "top_k": top_k}},
            )

            return results

        except Exception as e:
            logger.error(
                f"Dense retrieval failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return []

    async def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining dense and sparse methods.

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            List of retrieved documents
        """
        if not self._initialized:
            self.initialize()

        try:
            # For now, hybrid retrieval is the same as dense
            # In production, you would combine with BM25 or other sparse methods
            results = await self.retrieve_dense(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
            )

            logger.info(
                f"Hybrid retrieval completed: {len(results)} results",
                extra={"extra_fields": {"query": query[:100], "top_k": top_k}},
            )

            return results

        except Exception as e:
            logger.error(
                f"Hybrid retrieval failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return []

    async def retrieve_with_reranking(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 50,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieval with reranking for improved relevance.

        Args:
            query: Search query
            top_k: Final number of results
            initial_k: Initial retrieval size before reranking
            score_threshold: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            Reranked list of documents
        """
        if not self._initialized:
            self.initialize()

        try:
            # Retrieve more documents initially
            initial_results = await self.retrieve_dense(
                query=query,
                top_k=initial_k,
                score_threshold=score_threshold,
                filters=filters,
            )

            # For now, just return top_k results
            # In production, implement cross-encoder reranking
            results = initial_results[:top_k]

            logger.info(
                f"Retrieval with reranking completed: {len(results)} results",
                extra={"extra_fields": {"query": query[:100], "initial_k": initial_k}},
            )

            return results

        except Exception as e:
            logger.error(
                f"Retrieval with reranking failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return []

    async def retrieve_with_strategy(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using specified strategy.

        Args:
            query: Search query
            strategy: Retrieval strategy (dense, hybrid, reranking)
            top_k: Number of results
            score_threshold: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            Retrieved documents
        """
        if strategy == "dense":
            return await self.retrieve_dense(query, top_k, score_threshold, filters)
        elif strategy == "hybrid":
            return await self.retrieve_hybrid(query, top_k, score_threshold, filters)
        elif strategy == "reranking":
            return await self.retrieve_with_reranking(query, top_k, initial_k=top_k * 5, score_threshold=score_threshold, filters=filters)
        else:
            logger.warning(f"Unknown strategy '{strategy}', defaulting to hybrid")
            return await self.retrieve_hybrid(query, top_k, score_threshold, filters)


# Global instance
retrieval_strategy_manager = RetrievalStrategyManager()
