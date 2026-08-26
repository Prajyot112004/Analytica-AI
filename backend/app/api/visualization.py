from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.crud import (
    get_file_by_id,
    create_analysis_record,
    create_visualization_record,
    get_visualizations_by_session
)
from app.schemas.schemas import VisualizationRequest, VisualizationOut
from app.api.deps import get_current_user
from app.models.models import User
from app.agents.graph import analyst_agent

router = APIRouter(prefix="/visualizations", tags=["Visualizations"])

@router.post("/generate", response_model=VisualizationOut)
def generate_chart(req: VisualizationRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    file_obj = get_file_by_id(db, req.file_id)
    if not file_obj or file_obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="File not found")

    res_state = analyst_agent.process_query(
        user_id=current_user.id,
        session_id=req.session_id,
        query=f"Create a visualization: {req.query}",
        file_path=file_obj.file_path,
        file_type=file_obj.file_type,
        file_id=req.file_id
    )

    vis_res = res_state.get("visualization_result")
    if not vis_res:
        raise HTTPException(status_code=500, detail="Failed to generate visualization chart.")

    # Create Analysis record for the visualization
    analysis_rec = create_analysis_record(
        db,
        session_id=req.session_id,
        file_id=req.file_id,
        analysis_type="VISUALIZATION",
        request=f"Create visualization: {req.query}",
        code=vis_res.get("code"),
        result={"chart_type": vis_res["chart_type"], "image_path": vis_res["image_path"]},
        observations=vis_res.get("observations")
    )

    vis_rec = create_visualization_record(
        db,
        analysis_id=analysis_rec.id,
        chart_type=vis_res["chart_type"],
        title=vis_res["title"],
        image_path=vis_res["image_path"],
        code=vis_res.get("code"),
        observations=vis_res.get("observations")
    )

    return vis_rec


@router.get("", response_model=List[VisualizationOut])
def get_session_visualizations(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_visualizations_by_session(db, session_id=session_id)
