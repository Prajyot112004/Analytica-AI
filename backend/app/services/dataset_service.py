import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def load_dataset(file_path: str, file_type: str = "csv") -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ft = file_type.lower()
    if ft == "csv":
        # Try default comma delimiter, fallback to tab or semicolon if 1 col
        df = pd.read_csv(file_path)
        if df.shape[1] == 1:
            try:
                df_tab = pd.read_csv(file_path, sep="\t")
                if df_tab.shape[1] > 1:
                    df = df_tab
            except Exception:
                pass
        return df
    elif ft == "txt":
        try:
            return pd.read_csv(file_path, sep=r"\s+", engine="python")
        except Exception:
            return pd.read_csv(file_path, sep=",")
    elif ft == "json":
        try:
            return pd.read_json(file_path)
        except ValueError as e:
            raise ValueError(f"Failed to parse JSON file: {e}")
    else:
        raise ValueError(f"Unsupported file format: {file_type}")


def detect_outliers_iqr(series: pd.Series) -> int:
    if not pd.api.types.is_numeric_dtype(series) or series.dropna().empty:
        return 0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers)


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    row_count, col_count = df.shape
    columns_info = []
    data_types = {}
    missing_values = {}
    outliers_info = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()

    duplicates_count = int(df.duplicated().sum())

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        data_types[col] = dtype_str
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / row_count) * 100, 2) if row_count > 0 else 0.0
        missing_values[col] = {"count": null_count, "percentage": null_pct}

        n_unique = int(df[col].nunique())
        outlier_cnt = detect_outliers_iqr(df[col]) if col in numeric_cols else 0
        outliers_info[col] = outlier_cnt

        columns_info.append({
            "name": col,
            "dtype": dtype_str,
            "missing_count": null_count,
            "missing_pct": null_pct,
            "unique_count": n_unique,
            "is_numeric": col in numeric_cols,
            "is_categorical": col in categorical_cols,
            "outliers_count": outlier_cnt
        })

    # Summary Statistics
    stats_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        for col, row in desc.iterrows():
            stats_summary[col] = {
                "mean": round(float(row.get("mean", 0)), 4) if not pd.isna(row.get("mean")) else None,
                "std": round(float(row.get("std", 0)), 4) if not pd.isna(row.get("std")) else None,
                "min": float(row.get("min", 0)) if not pd.isna(row.get("min")) else None,
                "25%": float(row.get("25%", 0)) if not pd.isna(row.get("25%")) else None,
                "50%": float(row.get("50%", 0)) if not pd.isna(row.get("50%")) else None,
                "75%": float(row.get("75%", 0)) if not pd.isna(row.get("75%")) else None,
                "max": float(row.get("max", 0)) if not pd.isna(row.get("max")) else None,
            }

    # Target Detection heuristic
    potential_targets = []
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ["target", "label", "price", "sale", "revenue", "churn", "status", "outcome", "class"]):
            potential_targets.append(col)
    if not potential_targets and len(df.columns) > 0:
        potential_targets.append(df.columns[-1])

    profile = {
        "rows": row_count,
        "columns": col_count,
        "column_names": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "columns_info": columns_info,
        "data_types": data_types,
        "missing_values": missing_values,
        "duplicates_count": duplicates_count,
        "outliers_info": outliers_info,
        "statistics_summary": stats_summary,
        "potential_targets": potential_targets,
        "preview_data": df.head(10).replace({np.nan: None}).to_dict(orient="records")
    }

    return profile
