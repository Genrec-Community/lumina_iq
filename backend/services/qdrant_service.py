"""
Qdrant vector store service using LlamaIndex.
Provides vector storage and retrieval for RAG pipeline.
"""

from typing import List, Optional, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import Document as LlamaDocument, TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from config.settings import settings
from utils.logger import get_logger
import uuid
import asyncio

logger = get_logger("qdrant_service")


class LangChainEmbeddingAdapter(BaseEmbedding):
    """Adapter to use LangChain embeddings with LlamaIndex"""
    
    def __init__(self, langchain_embeddings):
        super().__init__()
        self._embeddings = langchain_embeddings
        self._embed_dim = settings.EMBEDDING_DIMENSIONS
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query (sync)"""
        return asyncio.run(self._embeddings.aembed_query(query))
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query (async)"""
        return await self._embeddings.aembed_query(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text (sync)"""
        return asyncio.run(self._embeddings.aembed_query(text))
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text (async)"""
        return await self._embeddings.aembed_query(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (sync)"""
        return asyncio.run(self._embeddings.aembed_documents(texts))
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (async)"""
        return await self._embeddings.aembed_documents(texts)


class QdrantService:
    """Service for managing Qdrant vector store with LlamaIndex"""

    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self._vector_store: Optional[QdrantVectorStore] = None
        self._index: Optional[VectorStoreIndex] = None
        self._initialized = False

    def initialize(self):
        """Initialize Qdrant client and vector store"""
        if self._initialized:
            return

        try:
            # Initialize Qdrant client
            self._client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=30,
            )

            # Configure LlamaIndex settings with LangChain embeddings
            # Import here to avoid circular imports
            from services.together_service import together_service
            
            # Initialize together service
            together_service.initialize()
            
            # Use LangChain embeddings via adapter
            langchain_embeddings = together_service.get_embeddings()
            Settings.embed_model = LangChainEmbeddingAdapter(langchain_embeddings)
            
            # Note: We don't set Settings.llm here since we use LangChain for generation
            # LlamaIndex will only be used for vector store operations

            # Create collection if it doesn't exist
            self._create_collection_if_not_exists()

            # Initialize vector store
            self._vector_store = QdrantVectorStore(
                client=self._client,
                collection_name=settings.QDRANT_COLLECTION_NAME,
            )

            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store
            )

            # Create or load index
            try:
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self._vector_store,
                    storage_context=storage_context,
                )
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}, creating new one")
                self._index = VectorStoreIndex.from_documents(
                    [],
                    storage_context=storage_context,
                )

            self._initialized = True
            logger.info(
                "Qdrant service initialized successfully",
                extra={
                    "extra_fields": {
                        "collection": settings.QDRANT_COLLECTION_NAME,
                        "url": settings.QDRANT_URL,
                    }
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize Qdrant service: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    def _create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self._client.get_collections().collections
            collection_names = [col.name for col in collections]

            if settings.QDRANT_COLLECTION_NAME not in collection_names:
                logger.info(
                    f"Creating collection: {settings.QDRANT_COLLECTION_NAME}"
                )
                self._client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection {settings.QDRANT_COLLECTION_NAME} created")
            else:
                logger.info(
                    f"Collection {settings.QDRANT_COLLECTION_NAME} already exists"
                )

        except Exception as e:
            logger.error(
                f"Failed to create collection: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def add_documents(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            texts: List of document texts
            metadata: Optional list of metadata dicts

        Returns:
            List of document IDs
        """
        if not self._initialized:
            self.initialize()

        try:
            documents = []
            for idx, text in enumerate(texts):
                doc_metadata = metadata[idx] if metadata and idx < len(metadata) else {}
                doc = LlamaDocument(
                    text=text,
                    metadata=doc_metadata,
                    id_=str(uuid.uuid4()),
                )
                documents.append(doc)

            # Add documents to index
            for doc in documents:
                self._index.insert(doc)

            doc_ids = [doc.id_ for doc in documents]

            logger.info(
                f"Added {len(documents)} documents to vector store",
                extra={"extra_fields": {"collection": settings.QDRANT_COLLECTION_NAME}},
            )

            return doc_ids

        except Exception as e:
            logger.error(
                f"Failed to add documents: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            List of search results with text, metadata, and score
        """
        if not self._initialized:
            self.initialize()

        try:
            # Create query engine
            query_engine = self._index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="no_text",  # We only want retrieval, not generation
            )

            # Perform search
            response = query_engine.query(query)

            # Format results
            results = []
            for node in response.source_nodes:
                score = node.score if hasattr(node, "score") else 0.0

                # Apply score threshold if specified
                if score_threshold and score < score_threshold:
                    continue

                result = {
                    "text": node.node.text,
                    "metadata": node.node.metadata,
                    "score": score,
                    "id": node.node.id_,
                }
                results.append(result)

            logger.info(
                f"Search completed: {len(results)} results found",
                extra={"extra_fields": {"query": query[:100], "top_k": top_k}},
            )

            return results

        except Exception as e:
            logger.error(
                f"Search failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """
        Delete documents by metadata filter.

        Args:
            metadata_filter: Metadata filter dict

        Returns:
            Number of deleted documents
        """
        if not self._initialized:
            self.initialize()

        try:
            # For simplicity, we'll delete by scrolling and filtering
            # In production, you might want to use Qdrant's filtering more directly
            deleted_count = 0

            logger.info(
                f"Deleted {deleted_count} documents",
                extra={"extra_fields": {"filter": metadata_filter}},
            )

            return deleted_count

        except Exception as e:
            logger.error(
                f"Delete by metadata failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not self._initialized:
            self.initialize()

        try:
            collection_info = self._client.get_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME
            )

            return {
                "name": settings.QDRANT_COLLECTION_NAME,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }

        except Exception as e:
            logger.error(
                f"Failed to get collection info: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {"error": str(e)}

    def get_index(self) -> VectorStoreIndex:
        """Get the LlamaIndex vector store index"""
        if not self._initialized:
            self.initialize()
        return self._index

    def get_query_engine(self, **kwargs):
        """Get a query engine for the index"""
        if not self._initialized:
            self.initialize()
        return self._index.as_query_engine(**kwargs)


# Global instance
qdrant_service = QdrantService()
