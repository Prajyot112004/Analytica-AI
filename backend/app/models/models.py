import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from app.db.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    files = relationship("File", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    files = relationship("File", back_populates="session", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="session", cascade="all, delete-orphan")
    ml_models = relationship("MLModel", back_populates="session", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "files"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # csv, txt
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="files")
    session = relationship("Session", back_populates="files")
    profile = relationship("DatasetProfile", back_populates="file", uselist=False, cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="file", cascade="all, delete-orphan")
    ml_models = relationship("MLModel", back_populates="file", cascade="all, delete-orphan")


class DatasetProfile(Base):
    __tablename__ = "dataset_profiles"

    id = Column(String, primary_key=True, default=generate_uuid)
    file_id = Column(String, ForeignKey("files.id"), nullable=False, unique=True)
    columns_info = Column(JSON, nullable=True)
    data_types = Column(JSON, nullable=True)
    missing_values = Column(JSON, nullable=True)
    duplicates_count = Column(Integer, default=0)
    statistics_summary = Column(JSON, nullable=True)
    profile_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    file = relationship("File", back_populates="profile")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    session = relationship("Session", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    file_id = Column(String, ForeignKey("files.id"), nullable=False)
    analysis_type = Column(String, nullable=False)  # CLEANING, EDA, VISUALIZATION, ML, RAG
    request = Column(Text, nullable=False)
    code = Column(Text, nullable=True)
    result = Column(JSON, nullable=True)
    observations = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="analyses")
    file = relationship("File", back_populates="analyses")
    visualizations = relationship("Visualization", back_populates="analysis", cascade="all, delete-orphan")


class Visualization(Base):
    __tablename__ = "visualizations"

    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=True)
    chart_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    code = Column(Text, nullable=True)
    image_path = Column(String, nullable=False)
    observations = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    analysis = relationship("Analysis", back_populates="visualizations")


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    file_id = Column(String, ForeignKey("files.id"), nullable=False)
    task_type = Column(String, nullable=False)  # Regression or Classification
    target_column = Column(String, nullable=False)
    algorithm = Column(String, nullable=False)
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=False)
    model_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="ml_models")
    file = relationship("File", back_populates="ml_models")


class Knowledge(Base):
    __tablename__ = "knowledge"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=True)
    source_type = Column(String, nullable=False)  # dataset, cleaning, eda, ml, chat
    content = Column(Text, nullable=False)
    embedding_id = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
