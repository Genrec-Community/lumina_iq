from datetime import datetime, timedelta
from fastapi import HTTPException
from config.settings import settings
from utils.storage import user_sessions
from utils.security import create_session_id
from models.auth import LoginRequest, LoginResponse
from utils.logger import get_logger

# Use enhanced logger
logger = get_logger("auth_service")


class AuthService:
    @staticmethod
    def login(request: LoginRequest) -> LoginResponse:
        logger.debug(f"Login attempt for user: {request.username}")

        if (
            request.username == settings.LOGIN_USERNAME
            and request.password == settings.LOGIN_PASSWORD
        ):
            logger.info("Login successful - credentials match")

            # Create a simple session ID for tracking (optional)
            session_id = create_session_id()
            user_sessions[session_id] = {
                "username": request.username,
                "created_at": datetime.now(),
                "expires_at": datetime.now()
                + timedelta(hours=settings.SESSION_EXPIRE_HOURS),
            }

            return LoginResponse(
                access_token=session_id,  # Return session ID but don't require it for API calls
                token_type="session",
                message="Login successful",
            )
        else:
            logger.warning("❌ Login failed - invalid credentials")
            raise HTTPException(status_code=401, detail="Invalid credentials")

    @staticmethod
    def logout() -> dict:
        # Simple logout - no token required
        logger.info("User logged out")
        return {"message": "Logout successful"}

    @staticmethod
    def is_valid_session(session_id: str = None) -> bool:
        # Optional session validation - only used if needed
        if not session_id:
            return True  # Allow access without strict session validation

        session = user_sessions.get(session_id)
        if session and session["expires_at"] > datetime.now():
            return True
        else:
            if session_id in user_sessions:
                del user_sessions[session_id]
            return False
