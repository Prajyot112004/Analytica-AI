from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.crud import (
    get_file_by_id, create_conversation, get_conversations_by_session,
    get_conversation_by_id, add_message, get_conversation_messages,
    create_analysis_record, create_visualization_record
)
from app.schemas.schemas import ChatMessageRequest, ChatMessageOut, ConversationOut
from app.api.deps import get_current_user
from app.models.models import User
from app.agents.graph import analyst_agent
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["AI Chatbot"])

class CreateConversationRequest(BaseModel):
    session_id: str
    title: str = "New Conversation"

@router.get("/conversations", response_model=List[ConversationOut])
def list_conversations(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_conversations_by_session(db, user_id=current_user.id, session_id=session_id)

@router.post("/conversations", response_model=ConversationOut)
def create_new_conversation(req: CreateConversationRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return create_conversation(db, user_id=current_user.id, session_id=req.session_id, title=req.title)

@router.post("/message", response_model=ChatMessageOut)
def send_chat_message(req: ChatMessageRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = get_conversation_by_id(db, user_id=current_user.id, conversation_id=req.conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save user message
    user_msg = add_message(db, conversation_id=conv.id, role="user", content=req.query)

    # Fetch file details if file_id provided
    file_path = None
    file_type = "csv"
    if req.file_id:
        file_obj = get_file_by_id(db, req.file_id)
        if file_obj:
            file_path = file_obj.file_path
            file_type = file_obj.file_type

    # Process through LangGraph Agent
    res_state = analyst_agent.process_query(
        user_id=current_user.id,
        session_id=req.session_id,
        query=req.query,
        file_path=file_path,
        file_type=file_type,
        file_id=req.file_id
    )

    bot_text = res_state.get("final_response", "I have processed your request.")

    # ── DB Persistance for Visualizations, ML, and Cleaning ──
    if req.file_id:
        intent = res_state.get("intent", "RAG_CHAT")
        
        # Save Visualization
        if res_state.get("visualization_result"):
            vis_res = res_state["visualization_result"]
            analysis_rec = create_analysis_record(
                db,
                session_id=req.session_id,
                file_id=req.file_id,
                analysis_type="VISUALIZATION",
                request=req.query,
                code=vis_res.get("code"),
                result=vis_res,
                observations=vis_res.get("observations")
            )
            create_visualization_record(
                db,
                chart_type=vis_res.get("chart_type", "bar"),
                title=vis_res.get("title", "Generated Chart"),
                image_path=vis_res.get("image_path"),
                analysis_id=analysis_rec.id,
                code=vis_res.get("code"),
                observations=vis_res.get("observations")
            )
            
        # Save Cleaning
        elif intent == "CLEANING" and res_state.get("execution_result"):
            create_analysis_record(
                db,
                session_id=req.session_id,
                file_id=req.file_id,
                analysis_type="CLEANING",
                request=req.query,
                result=res_state.get("execution_result"),
                observations=bot_text
            )
            
        # Save Machine Learning
        elif res_state.get("ml_result"):
            create_analysis_record(
                db,
                session_id=req.session_id,
                file_id=req.file_id,
                analysis_type="MACHINE_LEARNING",
                request=req.query,
                result=res_state.get("ml_result"),
                observations=bot_text
            )

    # Save assistant response
    assistant_msg = add_message(db, conversation_id=conv.id, role="assistant", content=bot_text)

    # Update conversation title if it's the first user message
    if conv.title == "New Conversation":
        conv.title = req.query[:30] + "..." if len(req.query) > 30 else req.query
        db.commit()

    return assistant_msg


@router.get("/history", response_model=List[ChatMessageOut])
def get_chat_history(conversation_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = get_conversation_by_id(db, user_id=current_user.id, conversation_id=conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return get_conversation_messages(db, conversation_id=conv.id)
