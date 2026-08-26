import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score


def train_ml_models(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    task_type: Optional[str] = None
) -> Dict[str, Any]:
    df_ml = df.copy().dropna(thresh=int(0.5 * len(df.columns)))  # Drop heavily empty rows

    # 1. Determine Target Column if not specified
    if not target_col or target_col not in df_ml.columns:
        potential_targets = [c for c in df_ml.columns if any(t in c.lower() for t in ["target", "label", "price", "sales", "revenue", "churn", "class", "status"])]
        target_col = potential_targets[0] if potential_targets else df_ml.columns[-1]

    # 2. Determine Task Type (Regression vs Classification)
    y_raw = df_ml[target_col]
    X_raw = df_ml.drop(columns=[target_col])

    if not task_type:
        if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 15:
            task_type = "Regression"
        else:
            task_type = "Classification"

    # Drop high-cardinality ID columns or raw text columns from X
    cols_to_drop = []
    for col in X_raw.columns:
        if X_raw[col].nunique() == len(X_raw) and not pd.api.types.is_numeric_dtype(X_raw[col]):
            cols_to_drop.append(col)
    if cols_to_drop:
        X_raw = X_raw.drop(columns=cols_to_drop)

    # 3. Preprocess Features (X)
    X_encoded = pd.DataFrame(index=X_raw.index)
    for col in X_raw.columns:
        if pd.api.types.is_numeric_dtype(X_raw[col]):
            imputer = SimpleImputer(strategy="median")
            X_encoded[col] = imputer.fit_transform(X_raw[[col]]).ravel()
        else:
            le = LabelEncoder()
            col_str = X_raw[col].astype(str).fillna("Missing")
            X_encoded[col] = le.fit_transform(col_str)

    # 4. Preprocess Target (y)
    if task_type == "Classification":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw.astype(str))
    else:
        imputer_y = SimpleImputer(strategy="mean")
        y = imputer_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel()

    # Train / Test Split
    if len(X_encoded) < 10:
        raise ValueError("Dataset has too few records for machine learning model training.")

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    best_model_name = ""
    best_score = -9999.0

    if task_type == "Regression":
        algorithms = {
            "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=50, random_state=42),
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
        }

        for name, algo in algorithms.items():
            try:
                algo.fit(X_train_scaled, y_train)
                y_pred = algo.predict(X_test_scaled)
                r2 = round(float(r2_score(y_test, y_pred)), 4)
                rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
                mae = round(float(mean_absolute_error(y_test, y_pred)), 4)

                results.append({
                    "algorithm": name,
                    "metrics": {
                        "R2_Score": r2,
                        "RMSE": rmse,
                        "MAE": mae
                    }
                })

                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
            except Exception as e:
                continue

    else:  # Classification
        algorithms = {
            "Random Forest Classifier": RandomForestClassifier(n_estimators=50, random_state=42),
            "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=50, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42)
        }

        for name, algo in algorithms.items():
            try:
                algo.fit(X_train_scaled, y_train)
                y_pred = algo.predict(X_test_scaled)
                acc = round(float(accuracy_score(y_test, y_pred)), 4)
                prec = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                rec = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)

                results.append({
                    "algorithm": name,
                    "metrics": {
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1_Score": f1
                    }
                })

                if acc > best_score:
                    best_score = acc
                    best_model_name = name
            except Exception as e:
                continue

    # Select best model
    best_result = next((r for r in results if r["algorithm"] == best_model_name), results[0] if results else None)

    summary = {
        "target_column": target_col,
        "task_type": task_type,
        "features_count": len(X_encoded.columns),
        "feature_names": list(X_encoded.columns),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_model": best_model_name,
        "all_model_results": results,
        "best_model_metrics": best_result["metrics"] if best_result else {}
    }

    return summary
