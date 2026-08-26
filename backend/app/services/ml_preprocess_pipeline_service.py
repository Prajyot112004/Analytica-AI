import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def cap_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Cap outliers using IQR bounds for numeric columns.
    Returns transformed DataFrame and action log.
    """
    actions = []
    df_capped = df.copy()
    numeric_cols = df_capped.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before_lower = (df_capped[col] < lower).sum()
        before_upper = (df_capped[col] > upper).sum()
        df_capped[col] = np.where(df_capped[col] < lower, lower, df_capped[col])
        df_capped[col] = np.where(df_capped[col] > upper, upper, df_capped[col])
        total = before_lower + before_upper
        if total > 0:
            actions.append(f"Capped {total} outlier(s) in numeric column '{col}' using IQR.")
    return df_capped, actions

def impute_missing(df: pd.DataFrame, strategy: str = 'median') -> Tuple[pd.DataFrame, List[str]]:
    """Impute missing values.
    Numeric columns use SimpleImputer with given strategy (median or mean).
    Categorical columns use most_frequent.
    Returns new DataFrame and actions.
    """
    actions = []
    df_imp = df.copy()
    # Numeric imputation
    num_cols = df_imp.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy=strategy)
        df_imp[num_cols] = imputer.fit_transform(df_imp[num_cols])
        actions.append(f"Imputed missing values in numeric columns using {strategy} strategy.")
    # Categorical imputation
    cat_cols = df_imp.select_dtypes(exclude=[np.number, 'datetime']).columns
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_imp[cat_cols] = cat_imputer.fit_transform(df_imp[cat_cols])
        actions.append("Imputed missing values in categorical columns using most frequent value.")
    return df_imp, actions

def encode_and_scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Encode categorical features and scale numeric features.
    Returns feature matrix (DataFrame) and actions.
    """
    actions = []
    df_enc = pd.DataFrame(index=df.index)
    # Encode categoricals
    cat_cols = df.select_dtypes(exclude=[np.number, 'datetime']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
        actions.append(f"Encoded categorical column '{col}' with LabelEncoder.")
    # Scale numerics
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df_enc[num_cols] = scaler.fit_transform(df[num_cols])
        actions.append("Scaled numeric columns using StandardScaler.")
    return df_enc, actions

def ml_preprocess_pipeline(df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Full ML preprocessing pipeline.
    Returns X (features DataFrame), y (target Series), and action log.
    If target_column is None, the last column is used as target.
    """
    actions = []
    df_ml = df.copy()
    # Determine target column
    if not target_column or target_column not in df_ml.columns:
        target_column = df_ml.columns[-1]
    y = df_ml[target_column]
    X = df_ml.drop(columns=[target_column])
    # Cap outliers
    X, cap_actions = cap_outliers(X)
    actions.extend(cap_actions)
    # Impute missing
    X, impute_actions = impute_missing(X)
    actions.extend(impute_actions)
    # Encode & scale
    X, enc_actions = encode_and_scale(X)
    actions.extend(enc_actions)
    return X, y, actions
