"""
Search service for document and semantic search.
"""

from typing import List, Dict, Any, Optional
from services.qdrant_service import qdrant_service
from services.embedding_service import embedding_service
from utils.logger import get_logger

logger = get_logger("search_service")


class SearchService:
    """Service for document search operations"""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Initialize the search service"""
        if not self._initialized:
            qdrant_service.initialize()
            self._initialized = True

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            List of search results
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
                f"Semantic search completed: {len(results)} results",
                extra={"extra_fields": {"query": query[:100]}},
            )

            return results

        except Exception as e:
            logger.error(
                f"Semantic search failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return []

    async def keyword_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search (simple implementation).

        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of results

        Returns:
            List of matching documents
        """
        try:
            query_lower = query.lower()
            query_terms = set(query_lower.split())

            results = []
            for idx, doc in enumerate(documents):
                doc_lower = doc.lower()
                doc_terms = set(doc_lower.split())

                # Calculate simple overlap score
                overlap = len(query_terms.intersection(doc_terms))
                if overlap > 0:
                    results.append({
                        "text": doc,
                        "score": overlap / len(query_terms),
                        "index": idx,
                    })

            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)

            return results[:top_k]

        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []


# Global instance
search_service = SearchService()
