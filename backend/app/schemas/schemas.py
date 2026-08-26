from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field

# User Schemas
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

# Session Schemas
class SessionCreate(BaseModel):
    name: str

class SessionOut(BaseModel):
    id: str
    user_id: str
    name: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# File & Dataset Schemas
class FileOut(BaseModel):
    id: str
    user_id: str
    session_id: str
    filename: str
    file_type: str
    file_size: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    uploaded_at: datetime

    class Config:
        from_attributes = True

class DatasetProfileOut(BaseModel):
    id: str
    file_id: str
    columns_info: Optional[List[Dict[str, Any]]] = None
    data_types: Optional[Dict[str, str]] = None
    missing_values: Optional[Dict[str, Any]] = None
    duplicates_count: Optional[int] = 0
    statistics_summary: Optional[Dict[str, Any]] = None
    profile_json: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Analysis & Cleaning Schemas
class CleaningRequest(BaseModel):
    session_id: str
    file_id: str

class AnalysisRequest(BaseModel):
    session_id: str
    file_id: str
    query: str

class AnalysisOut(BaseModel):
    id: str
    session_id: str
    file_id: str
    analysis_type: str
    request: str
    code: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    observations: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Visualization Schemas
class VisualizationRequest(BaseModel):
    session_id: str
    file_id: str
    query: str

class VisualizationOut(BaseModel):
    id: str
    chart_type: str
    title: str
    code: Optional[str] = None
    image_path: str
    observations: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# ML Schemas
class MLTrainRequest(BaseModel):
    session_id: str
    file_id: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # Regression or Classification

class MLModelOut(BaseModel):
    id: str
    session_id: str
    file_id: str
    task_type: str
    target_column: str
    algorithm: str
    metrics: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Chat Schemas
class ChatMessageRequest(BaseModel):
    session_id: str
    conversation_id: str
    file_id: Optional[str] = None
    query: str

class ConversationOut(BaseModel):
    id: str
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True
