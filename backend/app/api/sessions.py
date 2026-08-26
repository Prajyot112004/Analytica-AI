from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.crud import create_session, get_sessions_by_user, get_session_by_id
from app.schemas.schemas import SessionCreate, SessionOut
from app.api.deps import get_current_user
from app.models.models import User

router = APIRouter(prefix="/sessions", tags=["Sessions"])

@router.post("", response_model=SessionOut)
def create_new_session(session_in: SessionCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return create_session(db, user_id=current_user.id, name=session_in.name)

@router.get("", response_model=List[SessionOut])
def list_user_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = get_sessions_by_user(db, user_id=current_user.id)
    if not sessions:
        # Auto-create a default session if user has none
        default_sess = create_session(db, user_id=current_user.id, name="Default Analysis Workspace")
        return [default_sess]
    return sessions

@router.get("/{session_id}", response_model=SessionOut)
def get_session_details(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = get_session_by_id(db, session_id)
    if not sess or sess.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess
