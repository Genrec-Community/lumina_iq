from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import warning suppression first

from config.settings import settings
from routes import auth, pdf, chat
from utils.logger import get_logger
from utils.nltk_init import initialize_nltk_data
import asyncio
import platform

# Windows-compatible async optimizations
if platform.system() == "Windows":
    # Use ProactorEventLoop for better Windows performance
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
else:
    # Try uvloop on Unix systems
    try:
        import uvloop  # type: ignore

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Configure enhanced logging
    logger = get_logger("main")

    logger.info("Starting Learning App API...")
    logger.info("Initializing NLTK data...")
    initialize_nltk_data()
    yield
    # Shutdown
    logger.debug("Shutting down Learning App API...")


app = FastAPI(
    title="Learning App API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(pdf.router)
app.include_router(chat.router)


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Learning App API is running"}


# Additional health check endpoint
@app.head("/healthz")
async def healthz():
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        workers=1,  # Use 1 worker for development, increase for production
        loop="asyncio",  # Use asyncio (Windows compatible)
        access_log=True,
        log_level="debug",
        # Windows-compatible performance settings
        backlog=2048,
        timeout_keep_alive=5,
    )
