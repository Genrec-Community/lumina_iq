"""
LlamaIndex service wrapper for advanced RAG operations.
Provides LlamaIndex-specific functionality.
"""

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, Settings, Document as LlamaDocument
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from services.qdrant_service import qdrant_service
from utils.logger import get_logger

logger = get_logger("llamaindex_service")


class LlamaIndexService:
    """Wrapper service for LlamaIndex operations"""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Initialize LlamaIndex service"""
        if not self._initialized:
            qdrant_service.initialize()
            self._initialized = True

    def create_query_engine(
        self,
        similarity_top_k: int = 10,
        response_mode: str = "compact",
    ):
        """
        Create a query engine for RAG operations.

        Args:
            similarity_top_k: Number of similar documents to retrieve
            response_mode: Response synthesis mode

        Returns:
            Query engine instance
        """
        if not self._initialized:
            self.initialize()

        try:
            index = qdrant_service.get_index()

            # Create retriever
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=similarity_top_k,
            )

            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode(response_mode)
            )

            # Create query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )

            logger.info("Query engine created successfully")
            return query_engine

        except Exception as e:
            logger.error(f"Failed to create query engine: {str(e)}")
            raise

    async def query(
        self,
        query_text: str,
        similarity_top_k: int = 10,
        response_mode: str = "compact",
    ) -> Dict[str, Any]:
        """
        Query using LlamaIndex.

        Args:
            query_text: Query string
            similarity_top_k: Number of documents to retrieve
            response_mode: Response synthesis mode

        Returns:
            Query result with response and source nodes
        """
        if not self._initialized:
            self.initialize()

        try:
            query_engine = self.create_query_engine(
                similarity_top_k=similarity_top_k,
                response_mode=response_mode,
            )

            response = query_engine.query(query_text)

            # Format response
            result = {
                "response": str(response),
                "source_nodes": [
                    {
                        "text": node.node.text,
                        "score": node.score,
                        "metadata": node.node.metadata,
                    }
                    for node in response.source_nodes
                ],
            }

            logger.info(f"Query completed: {len(response.source_nodes)} sources used")
            return result

        except Exception as e:
            logger.error(f"LlamaIndex query failed: {str(e)}")
            return {
                "response": "Query failed",
                "error": str(e),
            }

    def get_retriever(
        self,
        similarity_top_k: int = 10,
    ):
        """Get a retriever instance"""
        if not self._initialized:
            self.initialize()

        index = qdrant_service.get_index()
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )


# Global instance
llamaindex_service = LlamaIndexService()
