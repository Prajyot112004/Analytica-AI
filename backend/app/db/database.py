from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.logging import logger

db_url = settings.DATABASE_URL

def get_engine(url: str):
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    try:
        engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        # Verify connection
        with engine.connect() as conn:
            pass
        logger.info(f"Successfully connected to database: {url.split('@')[-1] if '@' in url else url}")
        return engine
    except Exception as e:
        logger.warning(f"Could not connect to configured DATABASE_URL ({url}): {e}")
        # Fallback to local SQLite for seamless developer experience if PostgreSQL is offline
        fallback_url = "sqlite:///./analytica_fallback.db"
        logger.info(f"Falling back to local database: {fallback_url}")
        return create_engine(fallback_url, connect_args={"check_same_thread": False})

engine = get_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
