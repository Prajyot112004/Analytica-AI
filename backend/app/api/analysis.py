from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.crud import get_file_by_id, create_analysis_record, get_analyses_by_session, create_or_update_profile
from app.schemas.schemas import CleaningRequest, AnalysisRequest, AnalysisOut
from app.schemas.pipeline_models import PipelineRequest, PipelineOut, PipelineMode
from app.api.deps import get_current_user
from app.models.models import User
from app.agents.graph import analyst_agent
from app.services.dataset_service import load_dataset, profile_dataframe
from app.services.visualization_pipeline_service import visualization_pipeline
from app.services.ml_preprocess_pipeline_service import ml_preprocess_pipeline
from app.core.config import settings
import uuid
import os

router = APIRouter(prefix="/analysis", tags=["Analysis & Data Cleaning"])

@router.post("/clean", response_model=AnalysisOut)
def run_data_cleaning(req: CleaningRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    file_obj = get_file_by_id(db, req.file_id)
    if not file_obj or file_obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="File not found")

    res_state = analyst_agent.process_query(
        user_id=current_user.id,
        session_id=req.session_id,
        query="Clean my dataset",
        file_path=file_obj.file_path,
        file_type=file_obj.file_type,
        file_id=req.file_id
    )

    analysis_rec = create_analysis_record(
        db,
        session_id=req.session_id,
        file_id=req.file_id,
        analysis_type="CLEANING",
        request="Clean dataset",
        result=res_state.get("execution_result"),
        observations=res_state.get("final_response")
    )

    # Re-profile updated file if cleaned file was saved
    if res_state.get("execution_result") and "cleaned_file_path" in res_state["execution_result"]:
        cleaned_path = res_state["execution_result"]["cleaned_file_path"]
        file_obj.file_path = cleaned_path
        db.commit()
        
        cleaned_df = load_dataset(cleaned_path, file_obj.file_type)
        new_profile = profile_dataframe(cleaned_df)
        create_or_update_profile(
            db,
            file_id=req.file_id,
            columns_info=new_profile["columns_info"],
            data_types=new_profile["data_types"],
            missing_values=new_profile["missing_values"],
            duplicates_count=new_profile["duplicates_count"],
            statistics_summary=new_profile["statistics_summary"],
            profile_json=new_profile
        )

    return analysis_rec


@router.post("/custom", response_model=AnalysisOut)
def run_custom_analysis(req: AnalysisRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    file_obj = get_file_by_id(db, req.file_id)
    if not file_obj or file_obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="File not found")

    res_state = analyst_agent.process_query(
        user_id=current_user.id,
        session_id=req.session_id,
        query=req.query,
        file_path=file_obj.file_path,
        file_type=file_obj.file_type,
        file_id=req.file_id
    )

    analysis_rec = create_analysis_record(
        db,
        session_id=req.session_id,
        file_id=req.file_id,
        analysis_type=res_state.get("intent", "ANALYSIS"),
        request=req.query,
        code=res_state.get("generated_code"),
        result=res_state.get("execution_result") or res_state.get("visualization_result"),
        observations=res_state.get("final_response")
    )

    return analysis_rec


@router.get("", response_model=List[AnalysisOut])
def get_session_analyses(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_analyses_by_session(db, session_id=session_id)


@router.post("/pipeline", response_model=PipelineOut)
def run_pipeline(req: PipelineRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Load the dataset (cleaned version if exists, else original)
    file_obj = get_file_by_id(db, req.file_id)
    if not file_obj or file_obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="File not found")

    df = load_dataset(file_obj.file_path, file_obj.file_type)
    actions = []
    processed_path = None

    if req.mode == PipelineMode.VISUALIZATION:
        # Visualization pipeline: aggregation, flag outliers, format
        df_vis, vis_actions = visualization_pipeline(df, req.aggregation)
        actions.extend(vis_actions)
        # Save processed dataframe for possible downstream use
        filename = f"vis_processed_{uuid.uuid4().hex[:8]}.csv"
        processed_path = os.path.join(settings.GENERATED_PATH, filename)
        df_vis.to_csv(processed_path, index=False)
    else:  # ML mode
        X, y, ml_actions = ml_preprocess_pipeline(df, req.target_column)
        actions.extend(ml_actions)
        # Save features and target as separate files for downstream model training
        feats_file = f"ml_features_{uuid.uuid4().hex[:8]}.csv"
        target_file = f"ml_target_{uuid.uuid4().hex[:8]}.csv"
        feats_path = os.path.join(settings.GENERATED_PATH, feats_file)
        target_path = os.path.join(settings.GENERATED_PATH, target_file)
        X.to_csv(feats_path, index=False)
        y.to_frame(name=req.target_column or "target").to_csv(target_path, index=False)
        processed_path = {
            "features": feats_path,
            "target": target_path
        }

    # Record the pipeline execution
    analysis_rec = create_analysis_record(
        db,
        session_id=req.session_id,
        file_id=req.file_id,
        analysis_type="VISUALIZATION_PIPELINE" if req.mode == PipelineMode.VISUALIZATION else "ML_PIPELINE",
        request=f"Pipeline mode: {req.mode}",
        result={"processed_path": processed_path},
        observations="; ".join(actions) if actions else None,
        code=None
    )
    return PipelineOut(
        analysis_id=analysis_rec.id,
        mode=req.mode,
        actions_taken=actions,
        processed_path=processed_path if isinstance(processed_path, str) else None,
        details={"mode": req.mode}
    )
