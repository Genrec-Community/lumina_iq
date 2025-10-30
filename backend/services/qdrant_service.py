from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    MatchAny,
    Range,
)
from typing import List, Dict, Any, Optional
from config.settings import settings
from utils.logger import chat_logger
import hashlib
import uuid


class QdrantService:
    """Service for managing vector storage and retrieval using Qdrant Cloud"""

    def __init__(self):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            self.client = QdrantClient(
                url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=120
            )
            chat_logger.info("Qdrant client initialized successfully")
            self._ensure_collection_exists()
        except Exception as e:
            chat_logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise

    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if not"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSIONS,  # Dynamic embedding dimensions from settings
                        distance=Distance.COSINE,
                    ),
                )
                chat_logger.info("Created collection")
                chat_logger.debug(
                    f"Collection details: name={self.collection_name}, vector_size={settings.EMBEDDING_DIMENSIONS}, distance_metric=COSINE"
                )

                # Create payload indexes for filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="token",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="filename",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                # Create indexes for advanced metadata fields
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.chapter_number",
                    field_schema=PayloadSchemaType.INTEGER,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.section_number",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.sequential_id",
                    field_schema=PayloadSchemaType.INTEGER,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.primary_content_type",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                chat_logger.info("Created payload indexes")
                chat_logger.debug(
                    f"Payload indexes created for collection: {self.collection_name}"
                )
            else:
                chat_logger.info(f"Collection already exists: {self.collection_name}")
        except Exception as e:
            chat_logger.error(f"Failed to ensure collection exists: {str(e)}")
            raise

    def generate_chunk_id(self, filename: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk"""
        content = f"{filename}_{chunk_index}"
        hash_obj = hashlib.md5(content.encode())
        return hash_obj.hexdigest()

    async def index_document(
        self,
        filename: str,
        chunks: List[str],
        embeddings: List[List[float]],
        token: str,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """Index document chunks with their embeddings and rich metadata using batching"""
        try:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())

                # Get metadata for this chunk if available
                chunk_metadata = (
                    metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                )

                # Build payload with rich metadata
                payload = {
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,
                    "token": token,
                    "total_chunks": len(chunks),
                    "metadata": chunk_metadata,  # Store all rich metadata
                }

                point = PointStruct(id=point_id, vector=embedding, payload=payload)
                points.append(point)

            # Batch size for upsert operations (Qdrant has ~34MB payload limit)
            batch_size = 100
            total_batches = (
                len(points) + batch_size - 1
            ) // batch_size  # Ceiling division
            total_indexed = 0

            chat_logger.info(
                f"Starting batched indexing for {filename}: {len(points)} points in {total_batches} batches"
            )

            # Process points in batches
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(points))
                batch_points = points[start_idx:end_idx]

                try:
                    self.client.upsert(
                        collection_name=self.collection_name, points=batch_points
                    )

                    total_indexed += len(batch_points)
                    chat_logger.info(
                        f"Batch {batch_num + 1}/{total_batches}: Indexed {len(batch_points)} chunks "
                        f"(total: {total_indexed}/{len(points)})"
                    )

                except Exception as batch_error:
                    chat_logger.error(
                        f"Failed to index batch {batch_num + 1}/{total_batches} for {filename}: "
                        f"batch_size={len(batch_points)} "
                        f"start_idx={start_idx} "
                        f"end_idx={end_idx} "
                        f"error={str(batch_error)} "
                    )
                    # Continue with other batches even if one fails
                    continue

            chat_logger.info(
                f"Completed indexing {total_indexed} chunks for {filename} in {total_batches} batches"
            )
            return total_indexed

        except Exception as e:
            chat_logger.error(
                f"Failed to index document: filename={filename} error={str(e)}"
            )
            raise

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        token: str,
        filename: Optional[str] = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks based on query embedding with advanced filtering"""
        try:
            # Build filter
            filter_conditions = [
                FieldCondition(key="token", match=MatchValue(value=token))
            ]

            if filename:
                filter_conditions.append(
                    FieldCondition(key="filename", match=MatchValue(value=filename))
                )

            # Add advanced metadata filters
            if metadata_filters:
                for meta_filter in metadata_filters:
                    key = meta_filter.get("key")
                    value = meta_filter.get("value")
                    filter_type = meta_filter.get(
                        "type", "match"
                    )  # 'match', 'range', 'any'

                    # Build proper key path for nested metadata
                    if not key.startswith("metadata."):
                        key = f"metadata.{key}"

                    if filter_type == "match":
                        filter_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                    elif filter_type == "range":
                        filter_conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(gte=value.get("gte"), lte=value.get("lte")),
                            )
                        )

            query_filter = Filter(must=filter_conditions) if filter_conditions else None

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )

            results = []
            for hit in search_result:
                results.append(
                    {
                        "text": hit.payload.get("text", ""),
                        "score": hit.score,
                        "filename": hit.payload.get("filename", ""),
                        "chunk_index": hit.payload.get("chunk_index", 0),
                        "metadata": hit.payload.get("metadata", {}),
                    }
                )

            chat_logger.info(f"Found {len(results)} similar chunks")
            chat_logger.debug(
                f"Search details: token={token}, filename={filename}, limit={limit}"
            )
            return results
        except Exception as e:
            chat_logger.error("Failed to search similar chunks", error=str(e))
            raise

    async def delete_document_chunks(self, filename: str, token: str):
        """Delete all chunks for a specific document"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="filename", match=MatchValue(value=filename)
                        ),
                        FieldCondition(key="token", match=MatchValue(value=token)),
                    ]
                ),
            )
            chat_logger.info("Deleted chunks")
            chat_logger.debug(
                f"Deleted all chunks for document: filename={filename}, token={token}"
            )
        except Exception as e:
            chat_logger.error(
                f"Failed to delete document chunks: filename={filename}, token={token}, error={str(e)}"
            )
            raise

    async def check_document_indexed(self, filename: str, token: str) -> bool:
        """Check if a document is already indexed"""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="filename", match=MatchValue(value=filename)
                        ),
                        FieldCondition(key="token", match=MatchValue(value=token)),
                    ]
                ),
                limit=1,
            )

            return len(result[0]) > 0
        except Exception as e:
            chat_logger.error(
                "Failed to check if document is indexed: ",
                f"filename={filename}, error={str(e)} ",
            )
            return False

    async def get_chunks_by_filter(
        self,
        token: str,
        filename: str,
        metadata_filters: Optional[List[Dict[str, Any]]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get chunks by metadata filters (for notes generation)"""
        try:
            # Build filter
            filter_conditions = [
                FieldCondition(key="token", match=MatchValue(value=token)),
                FieldCondition(key="filename", match=MatchValue(value=filename)),
            ]

            # Add metadata filters
            if metadata_filters:
                for meta_filter in metadata_filters:
                    key = meta_filter.get("key")
                    value = meta_filter.get("value")

                    # Build proper key path
                    if not key.startswith("metadata."):
                        key = f"metadata.{key}"

                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

            query_filter = Filter(must=filter_conditions)

            # Use scroll to get all matching points
            result, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            chunks = []
            for point in result:
                chunks.append(
                    {
                        "text": point.payload.get("text", ""),
                        "score": 1.0,  # No score for filter-only retrieval
                        "filename": point.payload.get("filename", ""),
                        "chunk_index": point.payload.get("chunk_index", 0),
                        "metadata": point.payload.get("metadata", {}),
                    }
                )

            chat_logger.info("Retrieved chunks by filter")
            chat_logger.debug(
                f"Get chunks by filter details: token={token}, filename={filename}, filters={len(metadata_filters) if metadata_filters else 0}, limit={limit}"
            )
            return chunks

        except Exception as e:
            chat_logger.error(f"Failed to get chunks by filter: error={str(e)}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant service health and connectivity."""
        import time
        from datetime import datetime
        try:
            start_time = time.time()

            # Simple connectivity check - get collection info
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)

            latency_ms = (time.time() - start_time) * 1000

            return {
                "available": True,
                "collection_exists": collection_exists,
                "collection_name": self.collection_name,
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


qdrant_service = QdrantService()
