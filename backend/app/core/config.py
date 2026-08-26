import os
from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    PROJECT_NAME: str = "Analytica AI v2"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/analytica_db"

    # JWT Security
    SECRET_KEY: str = "analytica_secret_key_change_in_production_2026_super_secure"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Hugging Face API (Cloud Inference Only - No Local Heavy Models)
    HUGGINGFACEHUB_API_TOKEN: str = ""
    HF_LLM_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"
    HF_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    UPLOAD_DIR: str = "uploads"
    GENERATED_DIR: str = "generated"

    # CORS
    ALLOWED_CORS_ORIGINS: List[str] = ["*"]

    @property
    def UPLOAD_PATH(self) -> str:
        path = os.path.join(self.BASE_DIR, self.UPLOAD_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def GENERATED_PATH(self) -> str:
        path = os.path.join(self.BASE_DIR, self.GENERATED_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
        extra = "ignore"

settings = Settings()
