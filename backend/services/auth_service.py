"""
Authentication service for user login and session management.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("auth_service")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.JWT_SECRET if hasattr(settings, 'JWT_SECRET') else "change-this-in-production-use-at-least-32-chars"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = settings.SESSION_EXPIRE_HOURS


class AuthService:
    """Service for authentication and authorization"""

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """Decode and verify a JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.error(f"Token decode failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with username and password"""
        try:
            # Simple authentication against environment variables
            if username == settings.LOGIN_USERNAME and password == settings.LOGIN_PASSWORD:
                logger.info(f"User authenticated: {username}")
                return {
                    "username": username,
                    "authenticated": True,
                }
            else:
                logger.warning(f"Authentication failed for user: {username}")
                return None

        except Exception as e:
            logger.error(
                f"Authentication error: {str(e)}",
                extra={"extra_fields": {"error_type": type(e).__name__}},
            )
            return None

    @staticmethod
    async def login(username: str, password: str) -> Dict[str, Any]:
        """Login and generate access token"""
        user = await AuthService.authenticate_user(username, password)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token = AuthService.create_access_token(
            data={"sub": username}
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "message": f"Successfully logged in as {username}",
        }


# Global instance
auth_service = AuthService()
