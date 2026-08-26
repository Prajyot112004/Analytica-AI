from typing import Dict, Any, List, Optional, TypedDict

class AgentState(TypedDict):
    user_id: str
    session_id: str
    file_id: Optional[str]
    file_path: Optional[str]
    file_type: Optional[str]
    user_query: str
    dataset_profile: Optional[Dict[str, Any]]
    intent: Optional[str]  # CLEANING, ANALYSIS, VISUALIZATION, MACHINE_LEARNING, RAG_CHAT
    analysis_plan: Optional[List[str]]
    generated_code: Optional[str]
    execution_result: Optional[Any]
    visualization_result: Optional[Dict[str, Any]]
    ml_result: Optional[Dict[str, Any]]
    retrieved_context: Optional[str]
    final_response: Optional[str]
    error: Optional[str]
