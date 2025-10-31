"""
Chat service for handling user conversations and question generation.
Integrates RAG orchestrator with chat functionality.
"""

from typing import Dict, Any, List, Optional
from models.chat import (
    ChatMessage,
    ChatResponse,
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    QuizSubmissionRequest,
    QuizSubmissionResponse,
    QuizAnswer,
)
from services.rag_orchestrator import rag_orchestrator
from services.together_service import together_service
from utils.logger import get_logger
from utils.storage import chat_histories, pdf_contexts, storage_manager
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time

logger = get_logger("chat_service")

# Performance monitoring
request_times = deque(maxlen=100)
request_times_lock = threading.Lock()

# Thread pools for concurrent operations
ai_generation_pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="ai_gen")
model_creation_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="model_create")


class ChatService:
    """Service for chat operations"""

    @staticmethod
    async def chat(message: ChatMessage, user_session: str) -> ChatResponse:
        """
        Handle chat message from user.

        Args:
            message: Chat message from user
            user_session: User session ID

        Returns:
            Chat response with AI-generated answer
        """
        start_time = time.time()

        try:
            # Get chat history
            history = storage_manager.safe_get(chat_histories, user_session, [])

            # Get PDF context
            pdf_context = storage_manager.safe_get(pdf_contexts, user_session)

            if not pdf_context:
                response_text = "Please select a PDF document first before asking questions."
            else:
                filename = pdf_context.get("filename", "")

                # Ensure document is ingested
                await rag_orchestrator.ingest_pdf_for_user(user_session)

                # Use RAG orchestrator for retrieval and answer
                result = await rag_orchestrator.retrieve_and_answer(
                    query=message.message,
                    token=user_session,
                    filename=filename,
                )

                if result["status"] == "success":
                    response_text = result.get("answer", "I couldn't generate an answer.")
                else:
                    response_text = f"Error: {result.get('message', 'Unknown error')}"

            # Update chat history
            history.append({
                "role": "user",
                "content": message.message,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            })
            history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            })

            storage_manager.safe_set(chat_histories, user_session, history)

            # Track performance
            elapsed = time.time() - start_time
            with request_times_lock:
                request_times.append(elapsed)

            return ChatResponse(
                response=response_text,
                timestamp=datetime.datetime.utcnow().isoformat(),
            )

        except Exception as e:
            logger.error(
                f"Chat failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return ChatResponse(
                response=f"I encountered an error: {str(e)}",
                timestamp=datetime.datetime.utcnow().isoformat(),
            )

    @staticmethod
    async def generate_questions(
        user_session: str,
        topic: Optional[str] = None,
        count: int = 25,
        mode: str = "practice",
    ) -> ChatResponse:
        """
        Generate questions from PDF content.

        Args:
            user_session: User session ID
            topic: Optional topic to focus on
            count: Number of questions to generate
            mode: Question mode (quiz or practice)

        Returns:
            Chat response with generated questions
        """
        try:
            pdf_context = storage_manager.safe_get(pdf_contexts, user_session)

            if not pdf_context:
                return ChatResponse(
                    response="Please select a PDF document first.",
                    timestamp=datetime.datetime.utcnow().isoformat(),
                )

            filename = pdf_context.get("filename", "")

            # Use RAG orchestrator for question generation
            result = await rag_orchestrator.retrieve_and_generate_questions(
                query=topic or "comprehensive coverage",
                token=user_session,
                filename=filename,
                count=count,
                mode=mode,
            )

            if result["status"] == "success":
                response_text = result.get("response", "Failed to generate questions.")
            else:
                response_text = f"Error: {result.get('message', 'Unknown error')}"

            return ChatResponse(
                response=response_text,
                timestamp=datetime.datetime.utcnow().isoformat(),
            )

        except Exception as e:
            logger.error(
                f"Question generation failed: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return ChatResponse(
                response=f"Failed to generate questions: {str(e)}",
                timestamp=datetime.datetime.utcnow().isoformat(),
            )

    @staticmethod
    async def evaluate_answer(
        request: AnswerEvaluationRequest, user_session: str
    ) -> AnswerEvaluationResponse:
        """Evaluate a single answer"""
        try:
            prompt = f"""Evaluate the following answer to the question:

Question: {request.question}
User's Answer: {request.user_answer}

Provide:
1. A score out of 10
2. Detailed feedback
3. Suggestions for improvement
4. A hint about the correct answer (don't give it away completely)

Evaluation Level: {request.evaluation_level}

Respond in this format:
Score: [0-10]
Feedback: [Your feedback]
Suggestions: [Your suggestions]
Hint: [Your hint]"""

            response = await together_service.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=500,
            )

            # Parse response
            lines = response.strip().split("\n")
            score = 5
            feedback = "Good effort!"
            suggestions = "Keep practicing."
            hint = "Review the material again."

            for line in lines:
                if line.startswith("Score:"):
                    try:
                        score = int(line.split(":")[1].strip().split("/")[0])
                    except:
                        pass
                elif line.startswith("Feedback:"):
                    feedback = line.split(":", 1)[1].strip()
                elif line.startswith("Suggestions:"):
                    suggestions = line.split(":", 1)[1].strip()
                elif line.startswith("Hint:"):
                    hint = line.split(":", 1)[1].strip()

            return AnswerEvaluationResponse(
                question_id=request.question_id,
                score=score,
                max_score=10,
                feedback=feedback,
                suggestions=suggestions,
                correct_answer_hint=hint,
            )

        except Exception as e:
            logger.error(f"Answer evaluation failed: {str(e)}")
            return AnswerEvaluationResponse(
                question_id=request.question_id,
                score=0,
                max_score=10,
                feedback=f"Evaluation failed: {str(e)}",
                suggestions="Please try again.",
            )

    @staticmethod
    async def evaluate_quiz(
        request: QuizSubmissionRequest, user_session: str
    ) -> QuizSubmissionResponse:
        """Evaluate a complete quiz submission"""
        try:
            individual_results = []

            # Evaluate each answer
            for answer in request.answers:
                eval_request = AnswerEvaluationRequest(
                    question=answer.question,
                    user_answer=answer.user_answer,
                    question_id=answer.question_id,
                    evaluation_level=request.evaluation_level,
                )
                result = await ChatService.evaluate_answer(eval_request, user_session)
                individual_results.append(result)

            # Calculate overall score
            total_score = sum(r.score for r in individual_results)
            max_score = len(individual_results) * 10
            percentage = (total_score / max_score * 100) if max_score > 0 else 0

            # Assign grade
            if percentage >= 90:
                grade = "A"
            elif percentage >= 80:
                grade = "B"
            elif percentage >= 70:
                grade = "C"
            elif percentage >= 60:
                grade = "D"
            else:
                grade = "F"

            # Generate overall feedback
            overall_feedback = f"You scored {total_score}/{max_score} ({percentage:.1f}%). Grade: {grade}"

            # Identify strengths and areas for improvement
            strengths = []
            areas_for_improvement = []

            for result in individual_results:
                if result.score >= 8:
                    strengths.append(f"Q{result.question_id}: Excellent understanding")
                elif result.score <= 5:
                    areas_for_improvement.append(f"Q{result.question_id}: Needs more review")

            study_suggestions = [
                "Review the questions you scored below 7",
                "Focus on understanding concepts rather than memorization",
                "Practice more questions on weak areas",
            ]

            return QuizSubmissionResponse(
                overall_score=total_score,
                max_score=max_score,
                percentage=percentage,
                grade=grade,
                individual_results=individual_results,
                overall_feedback=overall_feedback,
                study_suggestions=study_suggestions,
                strengths=strengths or ["Keep up the good work!"],
                areas_for_improvement=areas_for_improvement or ["Continue practicing"],
            )

        except Exception as e:
            logger.error(f"Quiz evaluation failed: {str(e)}")
            raise

    @staticmethod
    def get_chat_history(user_session: str) -> Dict[str, Any]:
        """Get chat history for a user session"""
        history = storage_manager.safe_get(chat_histories, user_session, [])
        return {
            "history": history,
            "count": len(history),
        }

    @staticmethod
    def clear_chat_history(user_session: str) -> Dict[str, Any]:
        """Clear chat history for a user session"""
        storage_manager.safe_delete(chat_histories, user_session)
        return {
            "message": "Chat history cleared",
            "user_session": user_session,
        }
