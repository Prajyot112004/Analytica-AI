import os
# Load .env into os.environ FIRST — LangSmith SDK reads os.environ directly,
# not from pydantic Settings. Must happen before any langchain import.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger
from app.db.database import engine, Base
from app.api import auth, sessions, datasets, analysis, visualization, ml, chat

# Confirm LangSmith tracing status on startup
_tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
_project = os.environ.get("LANGCHAIN_PROJECT", "not set")
logger.info(
    f"LangSmith tracing: {'ENABLED' if _tracing_on else 'DISABLED'} "
    f"| project='{_project}'"
)

# Ensure database tables exist
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
except Exception as e:
    logger.warning(f"Could not create DB tables on startup: {e}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated chart images statically
app.mount("/generated", StaticFiles(directory=settings.GENERATED_PATH), name="generated")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_PATH), name="uploads")

# Include Routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(sessions.router, prefix=settings.API_V1_STR)
app.include_router(datasets.router, prefix=settings.API_V1_STR)
app.include_router(analysis.router, prefix=settings.API_V1_STR)
app.include_router(visualization.router, prefix=settings.API_V1_STR)
app.include_router(ml.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
