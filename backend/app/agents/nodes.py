import os
import pandas as pd
from typing import Dict, Any
from app.agents.state import AgentState
from app.agents.llm import invoke_llm
from app.services.dataset_service import load_dataset, profile_dataframe
from app.services.preprocessing_service import auto_clean_dataset
from app.services.visualization_service import generate_visualization
from app.services.ml_service import train_ml_models
from app.rag.retriever import retrieve_context_for_query, store_analysis_knowledge
from app.core.config import settings
from app.core.logging import logger
import json


def _build_dataset_summary(state: AgentState) -> str:
    """Load the active dataset and return a compact text summary for the LLM."""
    file_path = state.get("file_path")
    file_type = state.get("file_type", "csv")
    if not file_path or not os.path.exists(file_path):
        return ""
    try:
        df = load_dataset(file_path, file_type)
        profile = profile_dataframe(df)
        rows = profile["rows"]
        cols = profile["columns"]
        col_names = profile["column_names"]
        dtypes = profile["data_types"]
        missing = profile["missing_values"]
        stats = profile.get("statistics_summary", {})
        dupes = profile["duplicates_count"]

        lines = [
            f"Dataset: {os.path.basename(file_path)} ({rows} rows × {cols} columns)",
            f"Columns: {', '.join(col_names)}",
            f"Data types: {', '.join(f'{c}: {t}' for c, t in dtypes.items())}",
            f"Duplicates: {dupes}",
        ]
        # Missing values summary
        missing_cols = {c: v for c, v in missing.items() if v.get('count', 0) > 0}
        if missing_cols:
            lines.append("Missing values: " + ", ".join(
                f"{c}: {v['count']} ({v['percentage']}%)" for c, v in missing_cols.items()
            ))
        else:
            lines.append("Missing values: None")

        # Top statistics
        if stats:
            lines.append("Key statistics:")
            for col, s in list(stats.items())[:6]:  # limit to first 6 columns
                lines.append(f"  {col}: mean={s.get('mean')}, std={s.get('std')}, min={s.get('min')}, max={s.get('max')}")

        # Preview rows
        preview = profile.get("preview_data", [])
        if preview:
            lines.append(f"Sample rows (first 3):")
            for i, row in enumerate(preview[:3]):
                lines.append(f"  Row {i+1}: {row}")

        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to build dataset summary for LLM: {e}")
        return ""

def query_analyzer_node(state: AgentState) -> AgentState:
    query = state["user_query"].lower().strip()

    # Detect simple greetings / small talk first
    greeting_phrases = ["hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon",
                        "good evening", "what can you do", "help", "who are you", "what are you"]
    if any(query == g or query.startswith(g) for g in greeting_phrases):
        state["intent"] = "GREETING"
    elif any(k in query for k in ["clean", "null", "missing", "impute", "outlier", "duplicate", "fix data"]):
        state["intent"] = "CLEANING"
    elif any(k in query for k in ["chart", "plot", "graph", "visualize", "visualization", "bar", "hist", "scatter", "heatmap", "trend"]):
        state["intent"] = "VISUALIZATION"
    elif any(k in query for k in ["train", "model", "predict", "predictive", "algorithm", "accuracy", "r2", "regression", "classification", "random forest"]):
        state["intent"] = "MACHINE_LEARNING"
    elif any(k in query for k in ["analyze", "insight", "summary", "stats", "correlation", "distribution", "overview"]):
        state["intent"] = "ANALYSIS"
    else:
        state["intent"] = "RAG_CHAT"

    return state


def planner_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "RAG_CHAT")
    if intent == "GREETING":
        state["analysis_plan"] = ["1. Respond conversationally to the user greeting."]
    elif intent == "CLEANING":
        state["analysis_plan"] = [
            "1. Inspect missing values & data types",
            "2. Drop exact duplicate records",
            "3. Handle currency / percent symbols",
            "4. Apply mean/median/mode imputations",
            "5. Cap numerical outliers via IQR",
            "6. Generate cleaning summary report"
        ]
    elif intent == "VISUALIZATION":
        state["analysis_plan"] = [
            "1. Validate target & feature columns",
            "2. Select optimal Seaborn/Matplotlib chart type",
            "3. Render chart & save image artifact",
            "4. Formulate analytical chart observations"
        ]
    elif intent == "MACHINE_LEARNING":
        state["analysis_plan"] = [
            "1. Identify target variable & task type (Regression vs Classification)",
            "2. Preprocess features (imputation, encoding, scaling)",
            "3. Perform 80/20 train-test split",
            "4. Train 4 machine learning algorithms",
            "5. Evaluate metrics ($R^2$, RMSE, Accuracy, F1)",
            "6. Select best performing algorithm"
        ]
    elif intent == "ANALYSIS":
        state["analysis_plan"] = [
            "1. Profile dataset dimensions & data types",
            "2. Calculate univariate statistics & distributions",
            "3. Detect correlations & categorical insights",
            "4. Formulate actionable recommendations"
        ]
    else:
        state["analysis_plan"] = [
            "1. Retrieve knowledge context from RAG store",
            "2. Synthesize conversational response"
        ]

    return state


def cleaning_node(state: AgentState, df: pd.DataFrame, output_path: str) -> AgentState:
    cleaned_df, report = auto_clean_dataset(df, output_path)
    state["execution_result"] = report
    
    summary_text = (
        f"Data Cleaning Completed:\n"
        f"- Initial shape: {report['initial_shape'][0]} rows x {report['initial_shape'][1]} cols\n"
        f"- Final shape: {report['final_shape'][0]} rows x {report['final_shape'][1]} cols\n"
        f"- Actions applied:\n" + "\n".join([f"  * {act}" for act in report['actions_taken']])
    )
    state["final_response"] = summary_text

    # Persist in RAG Knowledge
    store_analysis_knowledge(
        session_id=state["session_id"],
        file_id=state.get("file_id", ""),
        knowledge_type="cleaning",
        content=summary_text
    )

    return state


def visualization_node(state: AgentState, df: pd.DataFrame) -> AgentState:
    query = state["user_query"]
    columns = list(df.columns)
    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
    
    prompt = f"""You are a data visualization expert.
Based on the following dataset schema and the user's query, determine the most appropriate chart type and the columns to plot.
User Query: "{query}"

Dataset Columns and Data Types:
{dtypes}

Available chart types: bar, line, histogram, scatter, box, heatmap.

Return ONLY a valid JSON object with no markdown formatting, no explanation, in this exact format:
{{
    "chart_type": "one of the available chart types",
    "x_col": "exact column name from the dataset, or null if not applicable",
    "y_col": "exact column name from the dataset, or null if not applicable"
}}
"""
    fallback_response = '{"chart_type": "bar", "x_col": null, "y_col": null}'
    
    llm_response = invoke_llm(prompt=prompt, fallback_response=fallback_response)
    
    try:
        clean_json = llm_response.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_json)
        chart_type = parsed.get("chart_type", "bar")
        x_col = parsed.get("x_col")
        y_col = parsed.get("y_col")
    except Exception:
        chart_type = "bar"
        x_col = None
        y_col = None

    if x_col not in columns: x_col = None
    if y_col not in columns: y_col = None

    vis_res = generate_visualization(df=df, chart_type=chart_type, x_col=x_col, y_col=y_col)
    state["visualization_result"] = vis_res
    
    summary_text = (
        f"Generated Visualization: {vis_res['title']}\n"
        f"Observation: {vis_res['observations']}"
    )
    state["final_response"] = summary_text

    # Persist in RAG Knowledge
    store_analysis_knowledge(
        session_id=state["session_id"],
        file_id=state.get("file_id", ""),
        knowledge_type="visualization",
        content=summary_text
    )

    return state


def ml_node(state: AgentState, df: pd.DataFrame) -> AgentState:
    try:
        ml_res = train_ml_models(df=df)
        state["ml_result"] = ml_res
        
        metrics_str = ", ".join([f"{k}: {v}" for k, v in ml_res["best_model_metrics"].items()])
        summary_text = (
            f"Machine Learning Training Results:\n"
            f"- Target Column: '{ml_res['target_column']}'\n"
            f"- Task Type: {ml_res['task_type']}\n"
            f"- Features Evaluated: {ml_res['features_count']}\n"
            f"- Best Algorithm: **{ml_res['best_model']}**\n"
            f"- Performance Metrics: {metrics_str}"
        )
        state["final_response"] = summary_text

        # Persist in RAG Knowledge
        store_analysis_knowledge(
            session_id=state["session_id"],
            file_id=state.get("file_id", ""),
            knowledge_type="machine_learning",
            content=summary_text
        )

    except Exception as e:
        state["error"] = str(e)
        state["final_response"] = f"Machine learning training encountered an error: {e}"

    return state


def rag_chat_node(state: AgentState) -> AgentState:
    context = retrieve_context_for_query(
        query=state["user_query"],
        session_id=state["session_id"]
    )
    state["retrieved_context"] = context

    context_block = context if context else "No prior analytical records found for this session."

    # ── Build dataset awareness block ──
    dataset_block = _build_dataset_summary(state)
    if dataset_block:
        dataset_section = f"Currently Active Dataset:\n{dataset_block}"
    else:
        dataset_section = "No dataset is currently loaded."

    prompt = (
        f"You are Analytica, a friendly and expert AI Data Analyst assistant.\n"
        f"Answer the user's question in a helpful, conversational way.\n"
        f"If the user is asking about data, use the dataset information and context below.\n"
        f"If the question is general (like a greeting), respond naturally.\n"
        f"IMPORTANT RULES:\n"
        f"- Do NOT generate or output markdown image tags (like ![img](data:image...)) or raw base64 strings.\n"
        f"- You cannot natively draw charts in this text response. If a user asks for a chart you cannot generate, explain that the system currently supports standard charts (bar, line, scatter, histogram, box, heatmap) via the visualization module, but not advanced ones like word clouds.\n\n"
        f"User Question: {state['user_query']}\n\n"
        f"{dataset_section}\n\n"
        f"Analytical Context (prior analyses, if relevant):\n{context_block}\n\n"
        f"Your response:"
    )

    # Friendly fallback
    if dataset_block:
        fallback = (
            f"I can see your active dataset! Here is a quick overview:\n{dataset_block}\n\n"
            "Feel free to ask me to clean, visualize, or analyze this data."
        )
    else:
        fallback = (
            "Hello! I'm Analytica, your AI Data Analyst. "
            "Upload a CSV, JSON, or TXT dataset and I can help you clean it, generate visualizations, "
            "train machine learning models, and answer questions about your data!"
        )

    response = invoke_llm(prompt=prompt, fallback_response=fallback)
    state["final_response"] = response
    return state
