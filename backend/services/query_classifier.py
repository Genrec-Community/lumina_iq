"""
Query Classifier Service
Classifies user queries into different use cases for optimal retrieval strategy
"""

from typing import Dict, Any, Tuple
import re
from utils.logger import chat_logger


class QueryClassifier:
    """
    Classify queries into use cases:
    - CHAT: General conversational Q&A
    - EVALUATION: Answer/quiz evaluation
    - QA_GENERATION: Question and answer generation
    - NOTES: Notes/summary generation
    """

    # Keywords for each use case
    USE_CASE_KEYWORDS = {
        "qa_generation": [
            "generate questions",
            "create questions",
            "make questions",
            "generate quiz",
            "create quiz",
            "make quiz",
            "generate q&a",
            "create q&a",
            "question generation",
            "generate mcq",
            "create mcq",
            "multiple choice",
            "generate test",
            "create test",
        ],
        "evaluation": [
            "evaluate",
            "check answer",
            "grade",
            "assess",
            "is this correct",
            "is this right",
            "verify answer",
            "check my answer",
            "correct answer",
            "validate",
            "score",
            "marks",
            "feedback on",
        ],
        "notes": [
            "generate notes",
            "create notes",
            "make notes",
            "summarize",
            "summary",
            "overview",
            "outline",
            "key points",
            "main points",
            "important points",
            "explain chapter",
            "explain section",
            "notes on",
            "give me notes",
        ],
        "chat": [
            "what is",
            "explain",
            "how does",
            "why",
            "tell me about",
            "describe",
            "define",
            "can you explain",
            "help me understand",
        ],
    }

    @staticmethod
    def classify_query(query: str) -> Dict[str, Any]:
        """
        Classify a query into a use case.

        Args:
            query: User's query text

        Returns:
            Dictionary with classification results:
            - use_case: The detected use case
            - confidence: Confidence score (0-1)
            - matched_keywords: Keywords that matched
        """
        query_lower = query.lower()

        # Track matches for each use case
        matches = {"qa_generation": [], "evaluation": [], "notes": [], "chat": []}

        # Check for keyword matches
        for use_case, keywords in QueryClassifier.USE_CASE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matches[use_case].append(keyword)

        # Calculate scores
        scores = {
            use_case: len(matched_keywords)
            for use_case, matched_keywords in matches.items()
        }

        # Determine use case with highest score
        if max(scores.values()) == 0:
            # No clear match, default to CHAT
            use_case = "chat"
            confidence = 0.5
            matched_keywords = []
        else:
            use_case = max(scores, key=scores.get)
            matched_keywords = matches[use_case]
            # Calculate confidence based on number of matches
            confidence = min(0.6 + (len(matched_keywords) * 0.15), 1.0)

        # Additional heuristics for better classification

        # Strong indicators for QA_GENERATION
        if re.search(r"\b\d+\s+questions?\b", query_lower):
            use_case = "qa_generation"
            confidence = 0.95

        # Strong indicators for EVALUATION
        if re.search(r"(my answer|the answer) (is|was)", query_lower):
            use_case = "evaluation"
            confidence = 0.90

        # Strong indicators for NOTES
        if re.search(
            r"(notes? (on|for|about)|summarize (chapter|section))", query_lower
        ):
            use_case = "notes"
            confidence = 0.90

        chat_logger.info("Query classified as")
        chat_logger.debug(
            f"Classification details: use_case={use_case}, confidence={confidence:.2f}, matched_keywords={matched_keywords}"
        )

        return {
            "use_case": use_case,
            "confidence": confidence,
            "matched_keywords": matched_keywords,
            "all_scores": scores,
        }

    @staticmethod
    def extract_context_requirements(query: str, use_case: str) -> Dict[str, Any]:
        """
        Extract context requirements based on use case.

        Returns:
            Dictionary with requirements:
            - chunk_size_preference: 'small', 'medium', 'large'
            - overlap_needed: bool
            - sequential_context: bool (need chunks in order)
            - num_chunks: suggested number of chunks to retrieve
        """
        requirements = {
            "chunk_size_preference": "medium",
            "overlap_needed": True,
            "sequential_context": False,
            "num_chunks": 5,
            "reranking_needed": True,
            "context_expansion": False,
        }

        if use_case == "qa_generation":
            requirements.update(
                {
                    "chunk_size_preference": "large",
                    "sequential_context": True,  # Need ordered chunks
                    "num_chunks": 15,
                    "reranking_needed": True,
                    "context_expansion": False,
                }
            )
            # Extract number of questions requested
            num_match = re.search(r"(\d+)\s+questions?", query.lower())
            if num_match:
                num_questions = int(num_match.group(1))
                requirements["num_chunks"] = max(num_questions, 15)

        elif use_case == "evaluation":
            requirements.update(
                {
                    "chunk_size_preference": "small",  # Precise matching
                    "sequential_context": False,
                    "num_chunks": 5,
                    "reranking_needed": True,
                    "context_expansion": True,  # Need surrounding context
                }
            )

        elif use_case == "notes":
            requirements.update(
                {
                    "chunk_size_preference": "large",
                    "sequential_context": True,  # Important for notes
                    "num_chunks": 20,
                    "reranking_needed": False,  # Order matters more than relevance
                    "context_expansion": False,
                }
            )

        elif use_case == "chat":
            requirements.update(
                {
                    "chunk_size_preference": "medium",
                    "sequential_context": False,
                    "num_chunks": 5,
                    "reranking_needed": True,
                    "context_expansion": False,
                }
            )

        return requirements


class QueryMetadataExtractor:
    """
    Extract metadata from queries:
    - Chapter numbers
    - Section numbers
    - Topic names
    - Difficulty levels
    """

    # Patterns for chapter extraction
    CHAPTER_PATTERNS = [
        r"\bchapter\s+(\d+)\b",
        r"\bch\.?\s*(\d+)\b",
        r"\bunit\s+(\d+)\b",
        r"\blesson\s+(\d+)\b",
        r"\bfrom chapter\s+(\d+)\b",
        r"\bin chapter\s+(\d+)\b",
    ]

    # Patterns for section extraction
    SECTION_PATTERNS = [
        r"\bsection\s+(\d+(?:\.\d+)?)\b",
        r"\bsec\.?\s*(\d+(?:\.\d+)?)\b",
    ]

    # Patterns for topic extraction
    TOPIC_PATTERNS = [
        r"(?:about|on|regarding|concerning)\s+(.+?)(?:\s+from|\s+in|\s+chapter|$)",
        r"(?:questions? (?:on|about))\s+(.+?)(?:\s+from|\s+in|\s+chapter|$)",
        r"(?:notes? (?:on|about))\s+(.+?)(?:\s+from|\s+in|\s+chapter|$)",
        r"(?:explain|describe|summarize)\s+(.+?)(?:\s+from|\s+in|\s+chapter|$)",
    ]

    @staticmethod
    def extract_chapter(query: str) -> Tuple[int, float]:
        """
        Extract chapter number from query.

        Returns:
            Tuple of (chapter_number, confidence) or (None, 0.0)
        """
        query_lower = query.lower()

        for pattern in QueryMetadataExtractor.CHAPTER_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    chapter_num = int(match.group(1))
                    # Higher confidence for explicit mentions
                    confidence = (
                        0.95
                        if "from chapter" in query_lower or "in chapter" in query_lower
                        else 0.85
                    )
                    return (chapter_num, confidence)
                except ValueError:
                    continue

        return (None, 0.0)

    @staticmethod
    def extract_section(query: str) -> Tuple[str, float]:
        """
        Extract section number from query.

        Returns:
            Tuple of (section_number, confidence) or (None, 0.0)
        """
        query_lower = query.lower()

        for pattern in QueryMetadataExtractor.SECTION_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                section_num = match.group(1)
                confidence = 0.90
                return (section_num, confidence)

        return (None, 0.0)

    @staticmethod
    def extract_topic(query: str) -> Tuple[str, float]:
        """
        Extract main topic from query.

        Returns:
            Tuple of (topic, confidence) or (None, 0.0)
        """
        query_lower = query.lower()

        for pattern in QueryMetadataExtractor.TOPIC_PATTERNS:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                # Clean up topic
                topic = re.sub(
                    r"\s+(from|in|chapter|section).*$", "", topic, flags=re.IGNORECASE
                )
                confidence = 0.80
                return (topic, confidence)

        # Fallback: use first few meaningful words
        words = [
            w
            for w in query.split()
            if len(w) > 3
            and w.lower()
            not in [
                "generate",
                "create",
                "make",
                "questions",
                "notes",
                "about",
                "from",
                "chapter",
            ]
        ]
        if words:
            topic = " ".join(words[:5])
            confidence = 0.50
            return (topic, confidence)

        return (None, 0.0)

    @staticmethod
    def extract_difficulty(query: str) -> Tuple[str, float]:
        """
        Extract difficulty level from query.

        Returns:
            Tuple of (difficulty, confidence) or ('medium', 0.5)
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ["easy", "simple", "basic", "beginner"]):
            return ("easy", 0.90)
        elif any(
            word in query_lower for word in ["hard", "difficult", "advanced", "complex"]
        ):
            return ("hard", 0.90)
        elif any(
            word in query_lower for word in ["medium", "moderate", "intermediate"]
        ):
            return ("medium", 0.90)

        # Default to medium
        return ("medium", 0.50)

    @staticmethod
    def extract_all_metadata(query: str) -> Dict[str, Any]:
        """
        Extract all metadata from query.

        Returns:
            Dictionary with all extracted metadata
        """
        chapter_num, chapter_conf = QueryMetadataExtractor.extract_chapter(query)
        section_num, section_conf = QueryMetadataExtractor.extract_section(query)
        topic, topic_conf = QueryMetadataExtractor.extract_topic(query)
        difficulty, diff_conf = QueryMetadataExtractor.extract_difficulty(query)

        metadata = {
            "chapter": {"value": chapter_num, "confidence": chapter_conf},
            "section": {"value": section_num, "confidence": section_conf},
            "topic": {"value": topic, "confidence": topic_conf},
            "difficulty": {"value": difficulty, "confidence": diff_conf},
        }

        chat_logger.info("Extracted query metadata")

        chat_logger.debug(
            f"Metadata details: chapter={chapter_num}({chapter_conf:.2f}), "
            f"section={section_num}({section_conf:.2f}), "
            f"topic={topic}({topic_conf:.2f}), "
            f"difficulty={difficulty}({diff_conf:.2f})"
        )

        return metadata


query_classifier = QueryClassifier()
query_metadata_extractor = QueryMetadataExtractor()
