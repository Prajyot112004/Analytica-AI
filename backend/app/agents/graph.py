import os
import pandas as pd
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.nodes import (
    query_analyzer_node,
    planner_node,
    cleaning_node,
    visualization_node,
    ml_node,
    rag_chat_node
)
from app.services.dataset_service import load_dataset
from app.core.config import settings

def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "RAG_CHAT")
    if intent == "CLEANING":
        return "cleaner"
    elif intent == "VISUALIZATION":
        return "visualizer"
    elif intent == "MACHINE_LEARNING":
        return "ml_agent"
    elif intent == "ANALYSIS":
        return "rag_chat"
    elif intent == "GREETING":
        return "rag_chat"  # handled conversationally
    else:
        return "rag_chat"


class DataAnalystWorkflow:
    def __init__(self):
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("query_analyzer", query_analyzer_node)
        workflow.add_node("planner", planner_node)
        workflow.add_node("cleaner", self.run_cleaner)
        workflow.add_node("visualizer", self.run_visualizer)
        workflow.add_node("ml_agent", self.run_ml_agent)
        workflow.add_node("rag_chat", rag_chat_node)

        # Set Entry Point
        workflow.set_entry_point("query_analyzer")
        workflow.add_edge("query_analyzer", "planner")

        # Conditional Edge Routing
        workflow.add_conditional_edges(
            "planner",
            route_intent,
            {
                "cleaner": "cleaner",
                "visualizer": "visualizer",
                "ml_agent": "ml_agent",
                "rag_chat": "rag_chat"
            }
        )

        workflow.add_edge("cleaner", END)
        workflow.add_edge("visualizer", END)
        workflow.add_edge("ml_agent", END)
        workflow.add_edge("rag_chat", END)

        self.app = workflow.compile()

    def run_cleaner(self, state: AgentState) -> AgentState:
        file_path = state.get("file_path")
        if not file_path or not os.path.exists(file_path):
            state["final_response"] = "No valid dataset uploaded for cleaning."
            return state

        df = load_dataset(file_path, state.get("file_type", "csv"))
        out_filename = f"cleaned_{os.path.basename(file_path)}"
        out_path = os.path.join(settings.UPLOAD_PATH, out_filename)
        return cleaning_node(state, df, out_path)

    def run_visualizer(self, state: AgentState) -> AgentState:
        file_path = state.get("file_path")
        if not file_path or not os.path.exists(file_path):
            state["final_response"] = "No valid dataset uploaded for visualization."
            return state

        df = load_dataset(file_path, state.get("file_type", "csv"))
        return visualization_node(state, df)

    def run_ml_agent(self, state: AgentState) -> AgentState:
        file_path = state.get("file_path")
        if not file_path or not os.path.exists(file_path):
            state["final_response"] = "No valid dataset uploaded for machine learning model training."
            return state

        df = load_dataset(file_path, state.get("file_type", "csv"))
        return ml_node(state, df)

    def process_query(
        self,
        user_id: str,
        session_id: str,
        query: str,
        file_path: str = None,
        file_type: str = "csv",
        file_id: str = None
    ) -> Dict[str, Any]:
        initial_state: AgentState = {
            "user_id": user_id,
            "session_id": session_id,
            "file_id": file_id,
            "file_path": file_path,
            "file_type": file_type,
            "user_query": query,
            "dataset_profile": None,
            "intent": None,
            "analysis_plan": None,
            "generated_code": None,
            "execution_result": None,
            "visualization_result": None,
            "ml_result": None,
            "retrieved_context": None,
            "final_response": None,
            "error": None
        }

        final_state = self.app.invoke(initial_state)
        return final_state

analyst_agent = DataAnalystWorkflow()
