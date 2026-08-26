from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session as DBSession
from app.models.models import (
    User, Session, File, DatasetProfile, Conversation,
    Message, Analysis, Visualization, MLModel, Knowledge
)
from app.core.security import get_password_hash

# User CRUD
def create_user(db: DBSession, username: str, email: str, password: str, full_name: Optional[str] = None) -> User:
    hashed_pwd = get_password_hash(password)
    user = User(
        username=username,
        email=email,
        password_hash=hashed_pwd,
        full_name=full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user_by_id(db: DBSession, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: DBSession, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: DBSession, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()

# Session CRUD
def create_session(db: DBSession, user_id: str, name: str) -> Session:
    session = Session(user_id=user_id, name=name)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def get_sessions_by_user(db: DBSession, user_id: str) -> List[Session]:
    return db.query(Session).filter(Session.user_id == user_id).order_by(Session.created_at.desc()).all()

def get_session_by_id(db: DBSession, session_id: str) -> Optional[Session]:
    return db.query(Session).filter(Session.id == session_id).first()

# File CRUD
def create_file_record(
    db: DBSession,
    user_id: str,
    session_id: str,
    filename: str,
    file_type: str,
    file_path: str,
    file_size: int,
    row_count: Optional[int] = None,
    column_count: Optional[int] = None
) -> File:
    file_obj = File(
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        file_size=file_size,
        row_count=row_count,
        column_count=column_count
    )
    db.add(file_obj)
    db.commit()
    db.refresh(file_obj)
    return file_obj

def get_file_by_id(db: DBSession, file_id: str) -> Optional[File]:
    return db.query(File).filter(File.id == file_id).first()

def get_files_by_session(db: DBSession, session_id: str) -> List[File]:
    return db.query(File).filter(File.session_id == session_id).order_by(File.uploaded_at.desc()).all()

# Profile CRUD
def create_or_update_profile(
    db: DBSession,
    file_id: str,
    columns_info: List[Dict[str, Any]],
    data_types: Dict[str, str],
    missing_values: Dict[str, Any],
    duplicates_count: int,
    statistics_summary: Dict[str, Any],
    profile_json: Dict[str, Any]
) -> DatasetProfile:
    existing = db.query(DatasetProfile).filter(DatasetProfile.file_id == file_id).first()
    if existing:
        existing.columns_info = columns_info
        existing.data_types = data_types
        existing.missing_values = missing_values
        existing.duplicates_count = duplicates_count
        existing.statistics_summary = statistics_summary
        existing.profile_json = profile_json
        db.commit()
        db.refresh(existing)
        return existing
    
    profile = DatasetProfile(
        file_id=file_id,
        columns_info=columns_info,
        data_types=data_types,
        missing_values=missing_values,
        duplicates_count=duplicates_count,
        statistics_summary=statistics_summary,
        profile_json=profile_json
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile

def get_profile_by_file_id(db: DBSession, file_id: str) -> Optional[DatasetProfile]:
    return db.query(DatasetProfile).filter(DatasetProfile.file_id == file_id).first()

# Conversation & Messages CRUD
def create_conversation(db: DBSession, user_id: str, session_id: str, title: str = "New Conversation") -> Conversation:
    conv = Conversation(user_id=user_id, session_id=session_id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

def get_conversations_by_session(db: DBSession, user_id: str, session_id: str) -> List[Conversation]:
    return db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.session_id == session_id
    ).order_by(Conversation.created_at.desc()).all()

def get_conversation_by_id(db: DBSession, user_id: str, conversation_id: str) -> Optional[Conversation]:
    return db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.id == conversation_id
    ).first()

def add_message(db: DBSession, conversation_id: str, role: str, content: str) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

def get_conversation_messages(db: DBSession, conversation_id: str) -> List[Message]:
    return db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at.asc()).all()

# Analysis CRUD
def create_analysis_record(
    db: DBSession,
    session_id: str,
    file_id: str,
    analysis_type: str,
    request: str,
    code: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    observations: Optional[str] = None
) -> Analysis:
    analysis = Analysis(
        session_id=session_id,
        file_id=file_id,
        analysis_type=analysis_type,
        request=request,
        code=code,
        result=result,
        observations=observations
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis

def get_analyses_by_session(db: DBSession, session_id: str) -> List[Analysis]:
    return db.query(Analysis).filter(Analysis.session_id == session_id).order_by(Analysis.created_at.desc()).all()

# Visualization CRUD
def create_visualization_record(
    db: DBSession,
    chart_type: str,
    title: str,
    image_path: str,
    analysis_id: Optional[str] = None,
    code: Optional[str] = None,
    observations: Optional[str] = None
) -> Visualization:
    vis = Visualization(
        analysis_id=analysis_id,
        chart_type=chart_type,
        title=title,
        code=code,
        image_path=image_path,
        observations=observations
    )
    db.add(vis)
    db.commit()
    db.refresh(vis)
    return vis

def get_visualizations_by_session(db: DBSession, session_id: str) -> List[Visualization]:
    vis_list = db.query(Visualization).join(Analysis, Visualization.analysis_id == Analysis.id)\
        .filter(Analysis.session_id == session_id).order_by(Visualization.created_at.desc()).all()
    if not vis_list:
        # Fallback: if existing records were created before analysis_id linking
        orphan_vis = db.query(Visualization).filter(Visualization.analysis_id.is_(None)).order_by(Visualization.created_at.desc()).all()
        if orphan_vis:
            return orphan_vis
    return vis_list

# ML Model CRUD
def create_ml_model_record(
    db: DBSession,
    session_id: str,
    file_id: str,
    task_type: str,
    target_column: str,
    algorithm: str,
    metrics: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None
) -> MLModel:
    ml_obj = MLModel(
        session_id=session_id,
        file_id=file_id,
        task_type=task_type,
        target_column=target_column,
        algorithm=algorithm,
        parameters=parameters,
        metrics=metrics,
        model_path=model_path
    )
    db.add(ml_obj)
    db.commit()
    db.refresh(ml_obj)
    return ml_obj

def get_ml_models_by_session(db: DBSession, session_id: str) -> List[MLModel]:
    return db.query(MLModel).filter(MLModel.session_id == session_id).order_by(MLModel.created_at.desc()).all()

# Knowledge CRUD
def create_knowledge_record(
    db: DBSession,
    user_id: str,
    source_type: str,
    content: str,
    session_id: Optional[str] = None,
    embedding_id: Optional[str] = None,
    metadata_json: Optional[Dict[str, Any]] = None
) -> Knowledge:
    k = Knowledge(
        user_id=user_id,
        session_id=session_id,
        source_type=source_type,
        content=content,
        embedding_id=embedding_id,
        metadata_json=metadata_json
    )
    db.add(k)
    db.commit()
    db.refresh(k)
    return k
