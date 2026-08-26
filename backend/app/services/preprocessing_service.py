import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    """
    Safe replacement for pd.to_numeric(series, errors='ignore').
    Pandas 2.2+ removed errors='ignore'. This replicates the old behaviour:
    - Values that convert successfully become numeric.
    - Values that fail conversion remain unchanged (original string value).
    """
    original = series.copy()
    converted = pd.to_numeric(series, errors='coerce')
    # Wherever conversion produced NaN but original was NOT NaN → restore original
    failed_mask = converted.isna() & original.notna()
    converted[failed_mask] = original[failed_mask]
    return converted


def auto_clean_dataset(df: pd.DataFrame, output_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_clean = df.copy()
    initial_rows, initial_cols = df_clean.shape
    actions_taken = []

    # 1. Drop exact Duplicate rows
    dup_count = int(df_clean.duplicated().sum())
    if dup_count > 0:
        df_clean.drop_duplicates(inplace=True)
        actions_taken.append(f"Removed {dup_count} duplicate row(s).")

    # 2. String trimming & symbol cleanup for object columns
    for col in df_clean.select_dtypes(include=['object']).columns:
        # Check if values look like currency or percentage (e.g., "$100", "25%")
        sample_vals = df_clean[col].dropna().astype(str).head(20)

        # Check for currency format
        if sample_vals.str.contains(r'^\s*[\$₹€£]').any():
            df_clean[col] = df_clean[col].astype(str).str.replace(r'[\$₹€£,]', '', regex=True)
            df_clean[col] = _to_numeric_safe(df_clean[col])
            actions_taken.append(f"Cleaned currency symbols and converted column '{col}' to numeric.")

        # Check for percentage format
        elif sample_vals.str.endswith('%').any():
            stripped = df_clean[col].astype(str).str.rstrip('%')
            converted = _to_numeric_safe(stripped)
            # Only divide where conversion actually succeeded (numeric dtype)
            if pd.api.types.is_numeric_dtype(converted):
                df_clean[col] = converted / 100.0
            else:
                df_clean[col] = converted
            actions_taken.append(f"Converted percentage column '{col}' to decimal numeric.")

    # 3. Detect and Standardize Timestamp / Date-Time columns
    for col in df_clean.select_dtypes(include=['object']).columns:
        col_lower = col.lower()
        has_date_hint = any(k in col_lower for k in ["date", "time", "timestamp", "datetime", "created_at", "updated_at", "dob", "joined", "year_month"])
        sample_vals = df_clean[col].dropna().astype(str).head(30)
        if sample_vals.empty:
            continue

        # If column name has date keyword OR sample values contain date-like separators (-, /, :)
        if has_date_hint or (sample_vals.str.contains(r'[-/:]').mean() > 0.6):
            try:
                parsed_dates = pd.to_datetime(sample_vals, errors='coerce', format='mixed')
                # Ensure it's not simply small numbers or plain text
                if parsed_dates.notna().mean() > 0.7 and not sample_vals.str.isnumeric().all():
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', format='mixed')
                    actions_taken.append(f"Detected and formatted column '{col}' into standardized datetime (YYYY-MM-DD HH:MM:SS).")
            except Exception:
                pass

    # 4. Handle missing values intelligently
    missing_before = int(df_clean.isnull().sum().sum())
    for col in df_clean.columns:
        null_count = df_clean[col].isnull().sum()
        if null_count == 0:
            continue

        null_pct = null_count / len(df_clean)

        # If column is > 70% missing, drop column
        if null_pct > 0.7:
            df_clean.drop(columns=[col], inplace=True)
            actions_taken.append(f"Dropped column '{col}' due to high missing rate ({round(null_pct*100, 1)}%).")
            continue

        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # Use median if skewed, mean if symmetrical
            skewness = df_clean[col].skew()
            if abs(skewness) > 1.0:
                fill_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_val)
                actions_taken.append(f"Imputed {null_count} missing value(s) in numeric column '{col}' with median ({round(fill_val, 4)}).")
            else:
                fill_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(fill_val)
                actions_taken.append(f"Imputed {null_count} missing value(s) in numeric column '{col}' with mean ({round(fill_val, 4)}).")
        else:
            # Categorical -> Mode imputation
            mode_val = df_clean[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(fill_val)
            actions_taken.append(f"Imputed {null_count} missing value(s) in categorical column '{col}' with mode ('{fill_val}').")

    # 4. Outlier handling via IQR capping
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        outlier_count = outliers_mask.sum()
        if outlier_count > 0:
            df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
            actions_taken.append(f"Capped {outlier_count} outlier(s) in numeric column '{col}' using IQR bounds.")

    # Save cleaned file
    df_clean.to_csv(output_path, index=False)

    final_rows, final_cols = df_clean.shape
    report = {
        "initial_shape": [initial_rows, initial_cols],
        "final_shape": [final_rows, final_cols],
        "rows_removed": initial_rows - final_rows,
        "cols_removed": initial_cols - final_cols,
        "missing_values_fixed": missing_before,
        "actions_taken": actions_taken,
        "cleaned_file_path": output_path
    }

    return df_clean, report
