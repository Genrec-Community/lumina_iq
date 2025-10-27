import socket
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_local_ip() -> str:
    """Get the local IP address for CORS configuration."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


class Settings(BaseSettings):
    # Load environment variables from backend/.env
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # For backwards compatibility with pydantic 1.x
    )

    # Authentication
    LOGIN_USERNAME: str = Field(default="vsbec")
    LOGIN_PASSWORD: str = Field(default="vsbec")

    # Gemini AI
    GEMINI_API_KEY: str = Field(default="AIzaSyBiKBADQGhRuFn5glEU-frmORFc0KRleVQ")
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash-lite")

    # Together.ai Configuration
    TOGETHER_API_KEY: str = Field(default="")
    TOGETHER_MODEL: str = Field(default="openai/gpt-oss-20b")
    TOGETHER_BASE_URL: str = Field(default="https://api.together.xyz/v1")

    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-large-en-v1.5")
    EMBEDDING_DIMENSIONS: int = Field(default=1024)

    # LlamaIndex Configuration for Render Free Tier
    LLAMAINDEX_CHUNK_SIZE: int = Field(default=512)  # Reduced for memory efficiency
    LLAMAINDEX_CHUNK_OVERLAP: int = Field(default=100)  # Reduced overlap
    LLAMAINDEX_USE_FOR_LARGE_PDFS: bool = Field(default=True)
    LLAMAINDEX_LARGE_PDF_THRESHOLD_MB: int = Field(default=10)

    # Qdrant Cloud Configuration
    QDRANT_URL: str = Field(
        default="https://1f6b3bbc-d09e-40c2-b333-0a823825f876.europe-west3-0.gcp.cloud.qdrant.io:6333"
    )
    QDRANT_API_KEY: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.O8xNwnZuHGOxo1dcIdcgKrRVZGryxKPYyGaCVyNXziQ"
    )
    QDRANT_COLLECTION_NAME: str = Field(default="learning_app_documents")
    QDRANT_USE_HYBRID_SEARCH: bool = Field(default=True)  # Enable hybrid search for better recall

    # CORS - Dynamic IP detection for development
    @property
    def CORS_ORIGINS(self) -> List[str]:
        local_ip = get_local_ip()
        origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
            f"http://{local_ip}:3000",
            f"http://{local_ip}:3001",
            "*",
        ]
        return list(set(origins))  # Remove duplicates

    # File paths - Use relative paths from project root
    @property
    def BOOKS_DIR(self) -> str:
        # Get the project root directory (parent of backend)
        backend_dir = Path(__file__).parent.parent
        project_root = backend_dir.parent
        books_dir = project_root / "books"
        return str(books_dir)

    CACHE_DIR: str = Field(default="cache")

    # Session
    SESSION_EXPIRE_HOURS: int = Field(default=24)

    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    # Performance Settings for Render Free Tier (512MB RAM, 1 CPU core)
    MAX_WORKERS: int = Field(default=1)  # Single-threaded for 1 CPU core
    MAX_CONCURRENT_REQUESTS: int = Field(default=10)  # Reduced to prevent memory overload
    EMBEDDING_BATCH_SIZE: int = Field(default=10)  # Batch embeddings to reduce API calls
    CACHE_EMBEDDINGS: bool = Field(default=True)  # Enable caching for embeddings
    CACHE_QUERY_RESULTS: bool = Field(default=True)  # Enable caching for query results
    CACHE_TTL_SECONDS: int = Field(default=3600)  # 1 hour cache TTL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Debug log to validate API key loading
        if self.TOGETHER_API_KEY:
            print(
                f"TOGETHER_API_KEY loaded successfully: {self.TOGETHER_API_KEY[:10]}..."
            )  # Masked for security
        else:
            print("WARNING: TOGETHER_API_KEY not loaded from .env file")


settings = Settings()
