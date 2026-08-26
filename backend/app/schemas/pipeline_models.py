from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class PipelineMode(str, Enum):
    VISUALIZATION = "visualization"
    ML = "ml"

class PipelineRequest(BaseModel):
    session_id: str
    file_id: str
    mode: PipelineMode
    aggregation: Optional[Dict[str, Any]] = None  # e.g., {'time_granularity': 'M'}
    target_column: Optional[str] = None

class PipelineOut(BaseModel):
    analysis_id: str
    mode: PipelineMode
    actions_taken: List[str]
    processed_path: Optional[str] = None  # Path to processed data (CSV or features file)
    details: Optional[Dict[str, Any]] = None
