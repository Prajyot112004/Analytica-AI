from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.crud import get_file_by_id, create_ml_model_record, get_ml_models_by_session
from app.schemas.schemas import MLTrainRequest, MLModelOut
from app.api.deps import get_current_user
from app.models.models import User
from app.agents.graph import analyst_agent

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

@router.post("/train", response_model=MLModelOut)
def train_machine_learning_model(req: MLTrainRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    file_obj = get_file_by_id(db, req.file_id)
    if not file_obj or file_obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="File not found")

    query_str = "Train a machine learning model"
    if req.target_column:
        query_str += f" to predict {req.target_column}"

    res_state = analyst_agent.process_query(
        user_id=current_user.id,
        session_id=req.session_id,
        query=query_str,
        file_path=file_obj.file_path,
        file_type=file_obj.file_type,
        file_id=req.file_id
    )

    ml_res = res_state.get("ml_result")
    if not ml_res:
        raise HTTPException(status_code=500, detail=res_state.get("error", "Failed to train machine learning models."))

    ml_rec = create_ml_model_record(
        db,
        session_id=req.session_id,
        file_id=req.file_id,
        task_type=ml_res["task_type"],
        target_column=ml_res["target_column"],
        algorithm=ml_res["best_model"],
        metrics=ml_res["best_model_metrics"],
        parameters={"all_results": ml_res["all_model_results"]}
    )

    return ml_rec


@router.get("/models", response_model=List[MLModelOut])
def get_session_ml_models(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_ml_models_by_session(db, session_id=session_id)
