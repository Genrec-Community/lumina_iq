"""
Question and Answer generation service.
Generates practice questions and quizzes from content.
"""

from typing import List, Dict, Any, Optional
from services.together_service import together_service
from utils.logger import get_logger

logger = get_logger("qa_generation_service")


class QAGenerationService:
    """Service for generating questions and answers"""

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Initialize the QA generation service"""
        if not self._initialized:
            together_service.initialize()
            self._initialized = True

    async def generate_mcq_questions(
        self,
        context: str,
        count: int = 10,
        difficulty: str = "medium",
    ) -> str:
        """Generate multiple choice questions"""
        if not self._initialized:
            self.initialize()

        prompt = f"""Based on the following content, generate {count} multiple-choice questions (MCQs).
Difficulty level: {difficulty}

Each question should have:
- 4 options (A, B, C, D)
- One correct answer
- Clear and unambiguous wording

Format:
Q1. [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]

Content:
{context[:3000]}

Generate {count} MCQ questions:"""

        try:
            questions = await together_service.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=2048,
            )
            return questions.strip()

        except Exception as e:
            logger.error(f"MCQ generation failed: {str(e)}")
            return "Failed to generate MCQ questions."

    async def generate_open_questions(
        self,
        context: str,
        count: int = 10,
        question_type: str = "analytical",
    ) -> str:
        """Generate open-ended questions"""
        if not self._initialized:
            self.initialize()

        prompt = f"""Based on the following content, generate {count} open-ended {question_type} questions.

These questions should:
- Encourage critical thinking
- Test understanding of concepts
- Be clear and specific

Format:
Q1. [Question text]

Content:
{context[:3000]}

Generate {count} questions:"""

        try:
            questions = await together_service.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=1024,
            )
            return questions.strip()

        except Exception as e:
            logger.error(f"Open question generation failed: {str(e)}")
            return "Failed to generate open-ended questions."

    async def generate_practice_set(
        self,
        context: str,
        mcq_count: int = 5,
        open_count: int = 5,
    ) -> Dict[str, str]:
        """Generate a mixed practice set"""
        if not self._initialized:
            self.initialize()

        try:
            mcq_questions = await self.generate_mcq_questions(context, mcq_count)
            open_questions = await self.generate_open_questions(context, open_count)

            return {
                "mcq_questions": mcq_questions,
                "open_questions": open_questions,
                "total_questions": mcq_count + open_count,
            }

        except Exception as e:
            logger.error(f"Practice set generation failed: {str(e)}")
            return {
                "mcq_questions": "Failed to generate MCQ questions.",
                "open_questions": "Failed to generate open-ended questions.",
                "total_questions": 0,
            }


# Global instance
qa_generation_service = QAGenerationService()
