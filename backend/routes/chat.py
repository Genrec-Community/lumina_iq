from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import hashlib
from models.chat import (
    ChatMessage,
    ChatResponse,
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    QuizSubmissionRequest,
    QuizSubmissionResponse,
    QuizAnswer,
)
from services.chat_service import ChatService
from services.rag_orchestrator import rag_orchestrator
from utils.logger import get_logger
from utils.logging_config import set_request_id, get_request_id, clear_request_id

logger = get_logger("chat_routes")

router = APIRouter(prefix="/api/chat", tags=["chat"])


def get_simple_user_id(request: Request) -> str:
    """
    Create a simple user ID based on client IP to isolate users.
    Each user gets their own chat history and PDF context.
    """
    client_ip = request.client.host if request.client else "unknown"
    # Create a simple hash of the IP for session isolation
    user_id = hashlib.md5(client_ip.encode()).hexdigest()[:12]
    return f"user_{user_id}"


class QuestionGenerationRequest(BaseModel):
    topic: Optional[str] = None
    count: Optional[int] = 25
    mode: Optional[str] = "practice"  # "quiz" for MCQ, "practice" for open-ended


@router.post("/", response_model=ChatResponse)
async def chat(message: ChatMessage, request: Request):
    """Send a message to the AI assistant about the selected PDF"""
    # Set request ID for tracing
    request_id = set_request_id()

    logger.info(
        "Chat request received",
        extra={"extra_fields": {
            "endpoint": "/api/chat",
            "method": "POST",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "content_length": request.headers.get("content-length", "0")
        }}
    )

    try:
        user_session = get_simple_user_id(request)
        result = await ChatService.chat(message, user_session)

        logger.info(
            "Chat request completed successfully",
            extra={"extra_fields": {
                "response_length": len(result.response) if result.response else 0,
                "timestamp": result.timestamp
            }}
        )

        return result
    except Exception as e:
        logger.error(
            "Chat request failed",
            extra={"extra_fields": {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }}
        )
        raise
    finally:
        clear_request_id()


@router.get("/history")
async def get_chat_history(request: Request):
    """Get the chat history for the current session"""
    user_session = get_simple_user_id(request)
    return ChatService.get_chat_history(user_session)


@router.delete("/history")
async def clear_chat_history(request: Request):
    """Clear the chat history for the current session"""
    user_session = get_simple_user_id(request)
    return ChatService.clear_chat_history(user_session)


@router.post("/generate-questions", response_model=ChatResponse)
async def generate_questions(request: QuestionGenerationRequest, http_request: Request):
    """Generate Q&A questions from the selected PDF content, optionally focused on a specific topic"""
    # Set request ID for tracing
    request_id = set_request_id()

    logger.info(
        "Question generation request received",
        extra={"extra_fields": {
            "endpoint": "/api/chat/generate-questions",
            "method": "POST",
            "topic": request.topic,
            "count": request.count,
            "mode": request.mode
        }}
    )

    try:
        user_session = get_simple_user_id(http_request)

        # Get PDF context to extract filename
        from utils.storage import pdf_contexts

        if user_session not in pdf_contexts:
            raise HTTPException(status_code=400, detail="No PDF selected. Please select a PDF first.")

        pdf_context = pdf_contexts[user_session]
        filename = pdf_context.get("filename", "")

        if not filename:
            raise HTTPException(status_code=400, detail="PDF context is invalid. Please select a PDF again.")

        # Use the new RAG orchestrator instead of direct ChatService call
        logger.info(f"Generating {request.count} questions for topic '{request.topic}' using RAG orchestrator")

        result = await rag_orchestrator.retrieve_and_generate_questions(
            query=request.topic or "comprehensive document coverage",
            token=user_session,
            filename=filename,
            count=request.count or 25,
            mode=request.mode or "practice"
        )

        if result["status"] == "success":
            response = ChatResponse(
                response=result["response"],
                timestamp=result.get("timestamp", "")
            )

            logger.info(
                "Question generation completed successfully",
                extra={"extra_fields": {
                    "response_length": len(result["response"]),
                    "filename": filename,
                    "questions_count": request.count
                }}
            )

            return response
        else:
            # Fallback to original ChatService if orchestrator fails
            logger.warning(
                "RAG orchestrator failed, falling back to ChatService",
                extra={"extra_fields": {
                    "orchestrator_error": result.get('message', 'Unknown error')
                }}
            )
            fallback_result = await ChatService.generate_questions(
                user_session, request.topic, request.count, request.mode
            )

            logger.info("Fallback question generation completed")

            return fallback_result
    except Exception as e:
        logger.error(
            "Question generation request failed",
            extra={"extra_fields": {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }}
        )
        raise
    finally:
        clear_request_id()


@router.post("/evaluate-answer", response_model=AnswerEvaluationResponse)
async def evaluate_answer(request: AnswerEvaluationRequest, http_request: Request):
    """Evaluate a single user answer using AI"""
    user_session = get_simple_user_id(http_request)
    return await ChatService.evaluate_answer(request, user_session)


@router.post("/evaluate-quiz", response_model=QuizSubmissionResponse)
async def evaluate_quiz(request: QuizSubmissionRequest, http_request: Request):
    """Evaluate a complete quiz submission with overall feedback"""
    user_session = get_simple_user_id(http_request)
    return await ChatService.evaluate_quiz(request, user_session)


@router.get("/performance-stats")
async def get_performance_stats():
    """Get current performance statistics for monitoring 25+ concurrent users"""
    from services.chat_service import (
        request_times,
        request_times_lock,
        ai_generation_pool,
        model_creation_pool,
    )
    import statistics

    with request_times_lock:
        if request_times:
            avg_time = statistics.mean(request_times)
            min_time = min(request_times)
            max_time = max(request_times)
            recent_requests = len(request_times)
        else:
            avg_time = min_time = max_time = recent_requests = 0

    # Get orchestrator health
    orchestrator_health = await rag_orchestrator.get_orchestrator_health()

    return {
        "performance": {
            "avg_response_time": round(avg_time, 2),
            "min_response_time": round(min_time, 2),
            "max_response_time": round(max_time, 2),
            "recent_requests": recent_requests,
        },
        "thread_pools": {
            "ai_generation_active": ai_generation_pool._threads,
            "ai_generation_max": ai_generation_pool._max_workers,
            "model_creation_active": model_creation_pool._threads,
            "model_creation_max": model_creation_pool._max_workers,
        },
        "orchestrator": orchestrator_health,
        "status": "production_ready_rag_orchestrator",
    }
