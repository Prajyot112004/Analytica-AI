import os
import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File as FastAPIFile, Form
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.crud import (
    create_file_record, get_files_by_session, get_file_by_id,
    create_or_update_profile, get_profile_by_file_id
)
from app.schemas.schemas import FileOut, DatasetProfileOut
from app.api.deps import get_current_user
from app.models.models import User, File, Visualization
from app.services.dataset_service import load_dataset, profile_dataframe
from app.core.config import settings

router = APIRouter(prefix="/files", tags=["Datasets"])

@router.post("/upload", response_model=FileOut)
async def upload_dataset_file(
    session_id: str = Form(...),
    file: UploadFile = FastAPIFile(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["csv", "txt", "json"]:
        raise HTTPException(status_code=400, detail="Supported file types: CSV, TXT, JSON.")

    # Save file on disk
    unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_PATH, unique_filename)
    
    contents = await file.read()
    file_size = len(contents)
    
    with open(file_path, "wb") as f:
        f.write(contents)

    # Load and Profile Dataset automatically
    try:
        df = load_dataset(file_path, file_type=ext)
        profile_data = profile_dataframe(df)
        row_cnt, col_cnt = df.shape
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset file: {str(e)}")

    # Store File Record
    file_record = create_file_record(
        db,
        user_id=current_user.id,
        session_id=session_id,
        filename=file.filename,
        file_type=ext,
        file_path=file_path,
        file_size=file_size,
        row_count=row_cnt,
        column_count=col_cnt
    )

    # Store Profile Record
    create_or_update_profile(
        db,
        file_id=file_record.id,
        columns_info=profile_data["columns_info"],
        data_types=profile_data["data_types"],
        missing_values=profile_data["missing_values"],
        duplicates_count=profile_data["duplicates_count"],
        statistics_summary=profile_data["statistics_summary"],
        profile_json=profile_data
    )

    return file_record


# ── New endpoints for the redesigned dashboard ──────────────────────────────

@router.get("/by-user", response_model=List[FileOut])
def list_user_files(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return every file uploaded by the current user (all sessions)."""
    return db.query(File).filter(File.user_id == current_user.id).order_by(File.uploaded_at.desc()).all()


@router.get("/active", response_model=FileOut)
def get_active_file(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the most-recently uploaded file for the current user."""
    file_obj = (
        db.query(File)
        .filter(File.user_id == current_user.id)
        .order_by(File.uploaded_at.desc())
        .first()
    )
    if not file_obj:
        raise HTTPException(status_code=404, detail="No uploaded files found.")
    return file_obj


@router.get("", response_model=List[FileOut])
def get_session_files(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_files_by_session(db, session_id=session_id)


@router.get("/{file_id}", response_model=FileOut)
def get_file_details(file_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    file_obj = get_file_by_id(db, file_id)
    if not file_obj or file_obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="File not found")
    return file_obj


@router.get("/{file_id}/profile", response_model=DatasetProfileOut)
def get_dataset_profile(file_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = get_profile_by_file_id(db, file_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Dataset profile not found")
    return profile


@router.get("/{file_id}/preview")
def preview_dataset_rows(file_id: str, limit: int = 20, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    file_obj = get_file_by_id(db, file_id)
    if not file_obj or file_obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="File not found")

    df = load_dataset(file_obj.file_path, file_obj.file_type)
    return {
        "columns": list(df.columns),
        "rows": df.head(limit).fillna("").to_dict(orient="records")
    }


@router.get("/{file_id}/visualizations")
def get_file_visualizations(file_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return all visualizations linked to analyses of the given file."""
    file_obj = get_file_by_id(db, file_id)
    if not file_obj or file_obj.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="File not found")

    from app.models.models import Analysis
    vis_rows = (
        db.query(Visualization)
        .join(Analysis, Analysis.id == Visualization.analysis_id)
        .filter(Analysis.file_id == file_id)
        .order_by(Visualization.created_at.desc())
        .all()
    )
    return [
        {
            "id": v.id,
            "chart_type": v.chart_type,
            "title": v.title,
            "image_path": v.image_path,
            "observations": v.observations,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in vis_rows
    ]
