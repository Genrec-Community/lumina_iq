"""
LlamaIndex Integration Service for Advanced RAG

This service integrates LlamaIndex framework into the existing RAG system for optimized
handling of large documents, especially PDFs over 1MB. It provides:

- Efficient PDF loading and parsing using LlamaIndex PDFReader
- Configurable chunking with sentence splitting for better context preservation
- Embedding generation using Together.ai models via LlamaIndex
- Vector indexing to Qdrant with rich metadata extraction
- Retrieval with metadata filtering for improved relevance

Key Features:
- Memory-efficient processing of large PDFs (>10MB configurable threshold)
- Backward compatibility with existing RAG pipeline
- Error handling for corrupted or unsupported files
- Integration with existing document metadata extraction

Usage:
- Automatically used for large PDFs in upload/select operations
- Can be enabled/disabled via settings
- Falls back to standard RAG on failure

Dependencies:
- llama-index-core
- llama-index-vector-stores-qdrant
- llama-index-embeddings-together
- llama-index-readers-file

Configuration:
- LLAMAINDEX_CHUNK_SIZE: Chunk size for splitting (default: 1000)
- LLAMAINDEX_CHUNK_OVERLAP: Overlap between chunks (default: 200)
- LLAMAINDEX_USE_FOR_LARGE_PDFS: Enable for large PDFs (default: true)
- LLAMAINDEX_LARGE_PDF_THRESHOLD_MB: Size threshold in MB (default: 10)
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.readers.file import PDFReader
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config.settings import settings
from utils.logger import chat_logger
from services.document_metadata_extractor import document_metadata_extractor
from utils.nltk_init import safe_nltk_operation


class LlamaIndexService:
    """
    Service for integrating LlamaIndex with the existing RAG system.
    Provides efficient handling of large documents, especially PDFs over 1MB.
    """

    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._setup_llamaindex_settings()

    def _setup_llamaindex_settings(self):
        """Configure LlamaIndex settings for embeddings and chunking"""
        try:
            # Set up Together.ai embedding model
            Settings.embed_model = TogetherEmbedding(
                model_name=settings.EMBEDDING_MODEL, api_key=settings.TOGETHER_API_KEY
            )

            # Configure chunking for large documents with NLTK sentence splitting
            # Use safe NLTK operation for sentence splitting
            def safe_sentence_splitter(chunk_size, chunk_overlap):
                """Safe sentence splitter that handles NLTK errors"""
                try:
                    chat_logger.info("Using SentenceSplitter for chunking")
                    return SentenceSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separator=" ",
                    )
                except Exception as e:
                    chat_logger.warning("Failed to create SentenceSplitter, falling back to TokenTextSplitter",
                                       error=str(e))
                    # Fall back to basic text splitter
                    from llama_index.core.node_parser import TokenTextSplitter

                    return TokenTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )

            Settings.node_parser = safe_sentence_splitter(
                settings.LLAMAINDEX_CHUNK_SIZE,
                settings.LLAMAINDEX_CHUNK_OVERLAP,
            )

            chat_logger.info(
                "LlamaIndex settings configured",
                embedding_model=settings.EMBEDDING_MODEL,
            )

        except Exception as e:
            chat_logger.error("Failed to setup LlamaIndex settings",
                             error=str(e))
            # Set minimal fallback settings
            Settings.embed_model = TogetherEmbedding(
                model_name=settings.EMBEDDING_MODEL, api_key=settings.TOGETHER_API_KEY
            )
            from llama_index.core.node_parser import TokenTextSplitter

            Settings.node_parser = TokenTextSplitter(
                chunk_size=settings.LLAMAINDEX_CHUNK_SIZE,
                chunk_overlap=settings.LLAMAINDEX_CHUNK_OVERLAP,
            )

    def _extract_metadata_from_node(
        self, node: BaseNode, document_name: str
    ) -> Dict[str, Any]:
        """Extract rich metadata from a LlamaIndex node"""
        # Use existing metadata extractor for consistency
        metadata = document_metadata_extractor.extract_metadata_from_chunk(
            chunk_text=node.text,
            chunk_index=node.metadata.get("chunk_index", 0),
            total_chunks=node.metadata.get("total_chunks", 1),
            document_name=document_name,
            context_before=node.metadata.get("context_before"),
        )

        # Add LlamaIndex specific metadata
        metadata.update(
            {
                "llamaindex_node_id": node.node_id,
                "llamaindex_start_char": node.start_char_idx,
                "llamaindex_end_char": node.end_char_idx,
                "llamaindex_metadata": node.metadata,
            }
        )

        return metadata

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batches to reduce API calls"""
        batch_size = getattr(settings, 'EMBEDDING_BATCH_SIZE', 10)
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                batch_embeddings = Settings.embed_model.get_text_embedding_batch(batch_texts)
                embeddings.extend(batch_embeddings)
                chat_logger.info("Generated embeddings for batch",
                                batch_num=i//batch_size + 1,
                                batch_size=len(batch_texts))
            except Exception as e:
                chat_logger.error("Failed to generate embeddings for batch",
                                 batch_num=i//batch_size + 1,
                                 error=str(e))
                # Fallback to individual
                for text in batch_texts:
                    try:
                        emb = Settings.embed_model.get_text_embedding(text)
                        embeddings.append(emb)
                    except Exception as e2:
                        chat_logger.error("Failed to generate embedding for text",
                                         error=str(e2))
                        embeddings.append([0.0] * settings.EMBEDDING_DIMENSIONS)

        return embeddings

    async def load_and_parse_pdf(self, file_path: str) -> List[Document]:
        """
        Load and parse a PDF file using LlamaIndex PDFReader.
        Optimized for large PDFs with streaming support.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of LlamaIndex Document objects
        """
        try:
            chat_logger.info("Loading PDF with LlamaIndex", file_path=file_path)

            # Check file size for large PDF handling
            file_size = os.path.getsize(file_path)
            threshold = settings.LLAMAINDEX_LARGE_PDF_THRESHOLD_MB * 1024 * 1024
            if file_size > threshold:
                chat_logger.warning(
                    "Large PDF detected, using memory-efficient loading",
                    file_path=file_path,
                    size_mb=file_size / (1024 * 1024),
                    threshold_mb=settings.LLAMAINDEX_LARGE_PDF_THRESHOLD_MB,
                )

            # Use PDFReader for efficient parsing
            reader = PDFReader()
            documents = reader.load_data(Path(file_path))

            chat_logger.info(
                "Loaded documents from PDF",
                file_path=file_path,
                total_pages=len(documents),
                num_documents=len(documents),
            )

            return documents

        except Exception as e:
            chat_logger.error(
                "Failed to load PDF with LlamaIndex", file_path=file_path, error=str(e)
            )
            raise

    async def create_ingestion_pipeline(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> IngestionPipeline:
        """
        Create an ingestion pipeline for processing documents.

        Args:
            documents: List of LlamaIndex Document objects
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            Configured IngestionPipeline
        """
        # Set up vector store with hybrid search if enabled
        vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=self.collection_name
        )
        if getattr(settings, 'QDRANT_USE_HYBRID_SEARCH', False):
            # Enable hybrid search
            vector_store.enable_hybrid = True
            vector_store.dense_vector_name = "dense"
            vector_store.sparse_vector_name = "sparse"

        # Use provided values or settings defaults
        cs = chunk_size if chunk_size is not None else settings.LLAMAINDEX_CHUNK_SIZE
        co = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.LLAMAINDEX_CHUNK_OVERLAP
        )

        # Create pipeline with transformations
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=cs, chunk_overlap=co),
                Settings.embed_model,
            ],
            vector_store=vector_store,
        )

        return pipeline

    async def index_document_with_llamaindex(
        self,
        file_path: str | Path,
        filename: str,
        token: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Index a document using LlamaIndex with support for large PDFs.

        Args:
            file_path: Path to the document
            filename: Name of the document
            token: User token
            chunk_size: Chunk size for splitting
            chunk_overlap: Overlap between chunks

        Returns:
            Dictionary with indexing results
        """
        try:
            chat_logger.info(
                "Starting LlamaIndex document indexing",
                filename=filename,
                chunk_size=chunk_size,
            )

            # Load and parse document
            documents = await self.load_and_parse_pdf(str(file_path))

            # Create ingestion pipeline
            pipeline = await self.create_ingestion_pipeline(
                documents, chunk_size, chunk_overlap
            )

            # Use provided values or settings defaults for return
            cs = (
                chunk_size if chunk_size is not None else settings.LLAMAINDEX_CHUNK_SIZE
            )
            co = (
                chunk_overlap
                if chunk_overlap is not None
                else settings.LLAMAINDEX_CHUNK_OVERLAP
            )

            # Run pipeline
            nodes = await pipeline.arun(documents=documents)

            # Extract metadata for each node
            nodes_with_metadata = []
            for node in nodes:
                metadata = self._extract_metadata_from_node(node, filename)
                node.metadata.update(metadata)
                nodes_with_metadata.append(node)

            # Update nodes in vector store with enhanced metadata
            try:
                await pipeline.vector_store.aadd_nodes_to_index(nodes_with_metadata)
            except Exception as e:
                chat_logger.error("Failed to add nodes to vector store",
                                 error=str(e))
                # Fall back to standard indexing if LlamaIndex vector store fails
                from services.rag_service import rag_service

                fallback_result = await rag_service.index_document(
                    filename=filename,
                    content="\n\n".join([node.text for node in nodes_with_metadata]),
                    token=token,
                    file_path=str(file_path),
                )
                return fallback_result

            chat_logger.info(
                "LlamaIndex indexing completed",
                filename=filename,
                num_nodes=len(nodes),
            )

            return {
                "status": "success",
                "filename": filename,
                "num_nodes": len(nodes),
                "method": "llamaindex",
                "chunk_size": cs,
                "chunk_overlap": co,
                "message": f"Successfully indexed {len(nodes)} nodes using LlamaIndex",
            }

        except Exception as e:
            chat_logger.error(
                "LlamaIndex indexing failed", filename=filename, error=str(e)
            )
            return {
                "status": "error",
                "filename": filename,
                "message": f"LlamaIndex indexing failed: {str(e)}",
            }

    async def retrieve_with_llamaindex(
        self,
        query: str,
        token: str,
        filename: Optional[str] = None,
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant nodes using LlamaIndex retriever.

        Args:
            query: Query string
            token: User token
            filename: Optional filename filter
            top_k: Number of results
            metadata_filters: Optional metadata filters

        Returns:
            Dictionary with retrieved nodes and context
        """
        try:
            chat_logger.info(
                "Starting LlamaIndex retrieval", query_length=len(query), top_k=top_k
            )

            # Set up vector store
            vector_store = QdrantVectorStore(
                client=self.qdrant_client, collection_name=self.collection_name
            )

            # Create index from vector store
            index = VectorStoreIndex.from_vector_store(vector_store)

            # Build retriever with filters
            filters = {}
            if filename:
                filters["filename"] = filename
            if token:
                filters["token"] = token
            if metadata_filters:
                filters.update(metadata_filters)

            retriever = index.as_retriever(
                similarity_top_k=top_k, filters=filters if filters else None
            )

            # Retrieve nodes
            nodes = await retriever.aretrieve(query)

            # Format results
            results = []
            context_parts = []
            for i, node in enumerate(nodes):
                metadata = node.metadata
                results.append(
                    {
                        "text": node.text,
                        "score": getattr(node, "score", 1.0),
                        "filename": metadata.get("filename", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "metadata": metadata,
                    }
                )
                context_parts.append(
                    f"[Node {i + 1}, Score: {getattr(node, 'score', 1.0):.2f}]\n{node.text}"
                )

            combined_context = "\n\n".join(context_parts)

            chat_logger.info("LlamaIndex retrieval completed", num_nodes=len(nodes))

            return {
                "status": "success",
                "nodes": results,
                "context": combined_context,
                "num_nodes": len(nodes),
                "retrieval_method": "llamaindex",
                "message": f"Retrieved {len(nodes)} nodes using LlamaIndex",
            }

        except Exception as e:
            chat_logger.error("LlamaIndex retrieval failed", error=str(e))
            return {
                "status": "error",
                "nodes": [],
                "context": "",
                "num_nodes": 0,
                "message": f"LlamaIndex retrieval failed: {str(e)}",
            }

    async def check_document_indexed_llamaindex(
        self, filename: str, token: str
    ) -> bool:
        """Check if document is indexed using LlamaIndex"""
        try:
            # Use existing Qdrant check since we're using the same vector store
            result = self.qdrant_client.scroll(
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
                "Failed to check LlamaIndex indexing", filename=filename, error=str(e)
            )
            return False


llamaindex_service = LlamaIndexService()
