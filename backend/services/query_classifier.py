"""
Query classifier service for RAG pipeline.
Classifies queries to determine the best retrieval strategy.
"""

from typing import Dict, Any, Literal
from services.together_service import together_service
from utils.logger import get_logger
import re

logger = get_logger("query_classifier")

QueryType = Literal["factual", "conceptual", "procedural", "analytical", "general"]


class QueryClassifier:
    """Classifies user queries to optimize retrieval strategy"""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Initialize the classifier"""
        if not self._initialized:
            together_service.initialize()
            self._initialized = True

    async def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify a query to determine its type and characteristics.

        Args:
            query: The user query to classify

        Returns:
            Classification result with query_type, complexity, and suggested_strategy
        """
        if not self._initialized:
            self.initialize()

        try:
            # Simple rule-based classification
            query_lower = query.lower()

            # Determine query type
            query_type = self._determine_query_type(query_lower)

            # Determine complexity
            complexity = self._determine_complexity(query)

            # Suggest retrieval strategy
            strategy = self._suggest_strategy(query_type, complexity)

            classification = {
                "query_type": query_type,
                "complexity": complexity,
                "suggested_strategy": strategy,
                "requires_context": self._requires_context(query_lower),
                "is_question": self._is_question(query_lower),
            }

            logger.debug(
                f"Query classified",
                extra={"extra_fields": {"classification": classification}},
            )

            return classification

        except Exception as e:
            logger.error(
                f"Query classification failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            # Return default classification
            return {
                "query_type": "general",
                "complexity": "medium",
                "suggested_strategy": "hybrid",
                "requires_context": True,
                "is_question": True,
            }

    def _determine_query_type(self, query: str) -> QueryType:
        """Determine the type of query"""
        # Factual queries
        if any(word in query for word in ["what is", "who is", "when did", "where is", "define"]):
            return "factual"

        # Conceptual queries
        if any(word in query for word in ["explain", "describe", "concept", "theory", "why"]):
            return "conceptual"

        # Procedural queries
        if any(word in query for word in ["how to", "steps", "process", "procedure", "guide"]):
            return "procedural"

        # Analytical queries
        if any(word in query for word in ["analyze", "compare", "contrast", "evaluate", "assess"]):
            return "analytical"

        return "general"

    def _determine_complexity(self, query: str) -> str:
        """Determine query complexity based on length and structure"""
        word_count = len(query.split())

        if word_count < 5:
            return "simple"
        elif word_count < 15:
            return "medium"
        else:
            return "complex"

    def _suggest_strategy(self, query_type: QueryType, complexity: str) -> str:
        """Suggest the best retrieval strategy based on classification"""
        # Complex queries benefit from hybrid search
        if complexity == "complex":
            return "hybrid"

        # Factual queries work well with dense retrieval
        if query_type == "factual":
            return "dense"

        # Conceptual and analytical queries benefit from hybrid
        if query_type in ["conceptual", "analytical"]:
            return "hybrid"

        # Default to hybrid for best results
        return "hybrid"

    def _requires_context(self, query: str) -> bool:
        """Check if query requires document context"""
        standalone_patterns = [
            "hello", "hi", "thanks", "thank you",
            "ok", "okay", "yes", "no"
        ]

        return not any(pattern in query for pattern in standalone_patterns)

    def _is_question(self, query: str) -> bool:
        """Check if the query is a question"""
        question_words = ["what", "who", "when", "where", "why", "how", "is", "are", "can", "could", "would"]
        return query.strip().endswith("?") or any(query.startswith(word) for word in question_words)


# Global instance
query_classifier = QueryClassifier()
