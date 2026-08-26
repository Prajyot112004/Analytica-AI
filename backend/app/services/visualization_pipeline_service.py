import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

def aggregate_for_visualization(df: pd.DataFrame, agg_spec: Dict[str, Any] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate data for visualization.
    agg_spec can contain keys like:
        - 'time_column': column name containing datetime
        - 'time_granularity': pandas offset alias e.g., 'M' for month
        - 'group_by': list of categorical columns to group by
        - 'agg_funcs': dict mapping column to aggregation function (e.g., 'mean')
    Returns aggregated DataFrame and list of actions taken.
    """
    actions = []
    if not agg_spec:
        return df, actions
    df_agg = df.copy()
    # Time-based aggregation
    time_col = agg_spec.get('time_column')
    time_gran = agg_spec.get('time_granularity')
    if time_col and time_gran:
        if pd.api.types.is_datetime64_any_dtype(df_agg[time_col]):
            df_agg[time_col] = df_agg[time_col].dt.to_period(time_gran).dt.to_timestamp()
            actions.append(f"Aggregated '{time_col}' to granularity '{time_gran}'.")
    # Categorical grouping
    group_by = agg_spec.get('group_by')
    agg_funcs = agg_spec.get('agg_funcs')
    if group_by and agg_funcs:
        grouped = df_agg.groupby(group_by).agg(agg_funcs).reset_index()
        df_agg = grouped
        actions.append(f"Grouped by {group_by} with aggregations {agg_funcs}.")
    return df_agg, actions

def flag_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add a boolean column 'is_outlier' per numeric column using IQR.
    The column records whether the row is an outlier in any numeric field.
    """
    actions = []
    outlier_flags = pd.Series(False, index=df.index)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_outliers = (df[col] < lower) | (df[col] > upper)
        outlier_flags = outlier_flags | col_outliers
    df = df.copy()
    df['is_outlier'] = outlier_flags
    actions.append("Flagged outliers with 'is_outlier' column (no removal).")
    return df, actions

def format_for_visuals(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Human‑readable formatting for dates and currencies.
    Dates are converted to ``Jan 2024`` style strings.
    Numeric columns that look like currency are rounded to 2 decimals and prefixed with $.
    """
    actions = []
    df_formatted = df.copy()
    # Date formatting
    for col in df_formatted.select_dtypes(include=['datetime64', 'datetime']).columns:
        df_formatted[col] = df_formatted[col].dt.strftime('%b %Y')
        actions.append(f"Formatted datetime column '{col}' to 'Mon YYYY' strings.")
    # Currency formatting – simple heuristic based on column name
    for col in df_formatted.select_dtypes(include=[np.number]).columns:
        if 'price' in col.lower() or 'amount' in col.lower() or 'cost' in col.lower():
            df_formatted[col] = df_formatted[col].apply(lambda x: f"${x:,.2f}")
            actions.append(f"Formatted numeric column '{col}' as currency strings.")
    return df_formatted, actions

def visualization_pipeline(df: pd.DataFrame, agg_spec: Dict[str, Any] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Full pipeline for visualization preparation.
    Returns processed DataFrame and list of descriptive actions.
    """
    actions: List[str] = []
    df_proc, agg_actions = aggregate_for_visualization(df, agg_spec)
    actions.extend(agg_actions)
    df_proc, outlier_actions = flag_outliers(df_proc)
    actions.extend(outlier_actions)
    df_proc, fmt_actions = format_for_visuals(df_proc)
    actions.extend(fmt_actions)
    return df_proc, actions
