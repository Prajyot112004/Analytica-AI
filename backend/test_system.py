import os
import sys
from app.db.database import engine, Base, SessionLocal
from app.models.models import User, Session as UserSession
from app.services.dataset_service import load_dataset, profile_dataframe
from app.services.preprocessing_service import auto_clean_dataset
from app.services.visualization_service import generate_visualization
from app.services.ml_service import train_ml_models
from app.agents.graph import analyst_agent

def run_system_verification():
    print("==================================================")
    print("VERIFYING ANALYTICA AI v2 BACKEND COMPONENTS")
    print("==================================================")

    # 1. Database Verification
    print("\n[1/6] Initializing Database Schema...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    print(" -> Database Schema Initialized OK.")

    # 2. Dataset Profiling
    test_csv = os.path.join(os.path.dirname(__file__), "test_sales.csv")
    print(f"\n[2/6] Loading and Profiling Dataset: test_sales.csv...")
    df = load_dataset(test_csv, "csv")
    profile = profile_dataframe(df)
    print(f" -> Profiled {profile['rows']} rows x {profile['columns']} columns.")
    print(f" -> Detected Numeric Columns: {profile['numeric_columns']}")
    print(f" -> Potential Targets: {profile['potential_targets']}")

    # 3. Preprocessing & Cleaning Engine
    print("\n[3/6] Running Data Cleaning Engine...")
    cleaned_path = os.path.join(os.path.dirname(__file__), "uploads", "cleaned_test_sales.csv")
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df_clean, report = auto_clean_dataset(df, cleaned_path)
    print(f" -> Actions applied: {len(report['actions_taken'])}")
    for act in report['actions_taken']:
        print(f"    * {act}")

    # 4. Visualization Engine
    print("\n[4/6] Rendering Seaborn Visualizations...")
    os.makedirs(os.path.join(os.path.dirname(__file__), "generated"), exist_ok=True)
    vis_res = generate_visualization(df_clean, chart_type="bar", x_col="Region", y_col="Revenue")
    print(f" -> Chart Created: '{vis_res['title']}' at {vis_res['image_path']}")
    print(f" -> Observation: {vis_res['observations']}")

    # 5. Machine Learning Pipeline
    print("\n[5/6] Training Machine Learning Models...")
    ml_res = train_ml_models(df_clean)
    print(f" -> Target Detected: '{ml_res['target_column']}' | Task Type: {ml_res['task_type']}")
    print(f" -> Best Algorithm: {ml_res['best_model']}")
    print(f" -> Best Model Metrics: {ml_res['best_model_metrics']}")

    # 6. Agent Workflow
    print("\n[6/6] Executing LangGraph Agent Workflow...")
    agent_res = analyst_agent.process_query(
        user_id="test_user",
        session_id="test_session",
        query="Clean dataset and show me a bar chart of sales",
        file_path=cleaned_path,
        file_type="csv"
    )
    print(f" -> Agent Intent Detected: {agent_res.get('intent')}")
    print(f" -> Agent Final Response:\n{agent_res.get('final_response')}")

    print("\n==================================================")
    print("SUCCESS: SYSTEM VERIFICATION COMPLETED CLEANLY!")
    print("==================================================")

if __name__ == "__main__":
    run_system_verification()
