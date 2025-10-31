"""
Main RAG service for document retrieval and answer generation.
"""

from typing import List, Dict, Any, Optional
from services.together_service import together_service
from services.qdrant_service import qdrant_service
from services.embedding_service import embedding_service
from services.chunking_service import chunking_service
from services.query_classifier import query_classifier
from services.retrieval_strategy_manager import retrieval_strategy_manager
from utils.logger import get_logger
import datetime

logger = get_logger("rag_service")


class RAGService:
    """Main RAG service for retrieval-augmented generation"""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Initialize all RAG components"""
        if not self._initialized:
            together_service.initialize()
            qdrant_service.initialize()
            query_classifier.initialize()
            retrieval_strategy_manager.initialize()
            self._initialized = True

    async def ingest_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.

        Args:
            text: Document text
            metadata: Optional metadata

        Returns:
            Ingestion result with document IDs
        """
        if not self._initialized:
            self.initialize()

        try:
            # Chunk the document
            chunks = chunking_service.chunk_text(text, metadata)

            # Extract texts and metadata
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_metadata = [chunk["metadata"] for chunk in chunks]

            # Add to vector store
            doc_ids = await qdrant_service.add_documents(
                texts=chunk_texts,
                metadata=chunk_metadata,
            )

            logger.info(
                f"Document ingested: {len(chunks)} chunks",
                extra={"extra_fields": {"metadata": metadata}},
            )

            return {
                "status": "success",
                "chunk_count": len(chunks),
                "document_ids": doc_ids,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(
                f"Document ingestion failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {
                "status": "error",
                "message": str(e),
            }

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            query_text: User query
            top_k: Number of results
            score_threshold: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            Query result with retrieved documents and generated answer
        """
        if not self._initialized:
            self.initialize()

        try:
            # Classify the query
            classification = await query_classifier.classify_query(query_text)

            # Retrieve relevant documents
            strategy = classification.get("suggested_strategy", "hybrid")
            documents = await retrieval_strategy_manager.retrieve_with_strategy(
                query=query_text,
                strategy=strategy,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
            )

            # Generate answer using retrieved context
            if documents:
                context = "\n\n".join([doc["text"] for doc in documents])
                answer = await self._generate_answer(query_text, context)
            else:
                answer = "I couldn't find relevant information in the documents to answer your question."

            return {
                "status": "success",
                "query": query_text,
                "answer": answer,
                "documents": documents,
                "classification": classification,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Query failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return {
                "status": "error",
                "message": str(e),
            }

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with context"""
        system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question accurately and concisely.
If the context doesn't contain enough information, acknowledge this and provide the best answer you can."""

        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        try:
            answer = await together_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1024,
            )
            return answer.strip()

        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "I apologize, but I encountered an error generating the answer."

    async def generate_questions(
        self,
        context: str,
        count: int = 25,
        mode: str = "practice",
    ) -> str:
        """
        Generate questions from context.

        Args:
            context: Document context
            count: Number of questions
            mode: Question mode (quiz or practice)

        Returns:
            Generated questions
        """
        if not self._initialized:
            self.initialize()

        try:
            if mode == "quiz":
                prompt = f"""Based on the following content, generate {count} multiple-choice questions (MCQs).
Each question should have 4 options (A, B, C, D) with one correct answer.
Format each question as:
Q1. [Question]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]

Content:
{context[:3000]}

Generate {count} MCQ questions:"""
            else:
                prompt = f"""Based on the following content, generate {count} open-ended practice questions.
These questions should test understanding and encourage critical thinking.
Format each question as:
Q1. [Question]

Content:
{context[:3000]}

Generate {count} practice questions:"""

            questions = await together_service.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=2048,
            )

            return questions.strip()

        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            return f"Failed to generate questions: {str(e)}"


# Global instance
rag_service = RAGService()
