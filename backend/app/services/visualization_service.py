import os
import uuid
import base64
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from app.core.config import settings

sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams['figure.facecolor'] = '#181b20'
plt.rcParams['axes.facecolor'] = '#21252b'
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['axes.labelcolor'] = '#cbd5e1'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['grid.color'] = '#334155'


def encode_image_to_base64(filepath: str) -> str:
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"


def generate_visualization(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 5.5))

    chart_type_lower = chart_type.lower()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    generated_code = ""
    observation = ""

    # Auto-fallback for x_col and y_col if not specified
    if not x_col and len(df.columns) > 0:
        x_col = categorical_cols[0] if categorical_cols else df.columns[0]
    if not y_col and len(numeric_cols) > 0:
        y_col = numeric_cols[0] if numeric_cols[0] != x_col else (numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])

    if "bar" in chart_type_lower:
        if x_col in categorical_cols and y_col in numeric_cols:
            grouped = df.groupby(x_col)[y_col].mean().reset_index().head(12)
            # Seaborn 0.13+: use hue= with legend=False instead of bare palette=
            sns.barplot(data=grouped, x=x_col, y=y_col, hue=x_col, ax=ax,
                        palette="mako", legend=False)
            ax.set_title(title or f"Average {y_col} by {x_col}", fontsize=14, color="#f8fafc", pad=15)
            generated_code = f"sns.barplot(data=df.groupby('{x_col}')['{y_col}'].mean().reset_index(), x='{x_col}', y='{y_col}', hue='{x_col}', legend=False)"
            observation = f"Bar chart showing average '{y_col}' aggregated across categories of '{x_col}'."
        else:
            # countplot: use hue= with legend=False
            top_vals = df[x_col].value_counts().iloc[:10].index
            sns.countplot(data=df, x=x_col, hue=x_col, ax=ax,
                          palette="viridis", order=top_vals, legend=False)
            ax.set_title(title or f"Count Distribution of {x_col}", fontsize=14, color="#f8fafc", pad=15)
            generated_code = f"sns.countplot(data=df, x='{x_col}', hue='{x_col}', legend=False)"
            observation = f"Distribution count of unique values in categorical feature '{x_col}'."

    elif "line" in chart_type_lower:
        sns.lineplot(data=df, x=x_col, y=y_col, ax=ax, color="#38bdf8", linewidth=2.5, marker="o")
        ax.set_title(title or f"Trend of {y_col} over {x_col}", fontsize=14, color="#f8fafc", pad=15)
        generated_code = f"sns.lineplot(data=df, x='{x_col}', y='{y_col}')"
        observation = f"Line plot illustrating sequential relationship and trends between '{x_col}' and '{y_col}'."

    elif "hist" in chart_type_lower or "dist" in chart_type_lower:
        col_to_plot = y_col if y_col in numeric_cols else (x_col if x_col in numeric_cols else (numeric_cols[0] if numeric_cols else df.columns[0]))
        sns.histplot(data=df, x=col_to_plot, kde=True, ax=ax, color="#818cf8", bins=20)
        ax.set_title(title or f"Distribution Histogram of {col_to_plot}", fontsize=14, color="#f8fafc", pad=15)
        generated_code = f"sns.histplot(data=df, x='{col_to_plot}', kde=True)"
        observation = f"Distribution density and histogram showing skewness and variance for feature '{col_to_plot}'."

    elif "scatter" in chart_type_lower:
        if len(numeric_cols) >= 2:
            num_x = x_col if x_col in numeric_cols else numeric_cols[0]
            num_y = y_col if y_col in numeric_cols else numeric_cols[1]
            sns.scatterplot(data=df, x=num_x, y=num_y, ax=ax, color="#f43f5e", alpha=0.8, s=60)
            ax.set_title(title or f"Scatter Plot: {num_x} vs {num_y}", fontsize=14, color="#f8fafc", pad=15)
            generated_code = f"sns.scatterplot(data=df, x='{num_x}', y='{num_y}')"
            corr_val = df[[num_x, num_y]].corr().iloc[0, 1]
            observation = f"Scatter plot of '{num_x}' versus '{num_y}' (Pearson Correlation coefficient: {round(corr_val, 3)})."
        else:
            sns.histplot(data=df, x=df.columns[0], kde=True, ax=ax)
            generated_code = f"sns.histplot(data=df, x='{df.columns[0]}')"
            observation = "Fallback histogram generated."

    elif "box" in chart_type_lower:
        if x_col in categorical_cols and y_col in numeric_cols:
            # Seaborn 0.13+: use hue= with legend=False instead of bare palette=
            sns.boxplot(data=df, x=x_col, y=y_col, hue=x_col, ax=ax,
                        palette="crest", legend=False)
            ax.set_title(title or f"Box Plot of {y_col} grouped by {x_col}", fontsize=14, color="#f8fafc", pad=15)
            generated_code = f"sns.boxplot(data=df, x='{x_col}', y='{y_col}', hue='{x_col}', legend=False)"
            observation = f"Box plot showing median, quartiles, and outliers for '{y_col}' across categories of '{x_col}'."
        else:
            num_col = y_col if y_col in numeric_cols else (numeric_cols[0] if numeric_cols else df.columns[0])
            sns.boxplot(data=df, y=num_col, ax=ax, color="#a855f7")
            ax.set_title(title or f"Box Plot of {num_col}", fontsize=14, color="#f8fafc", pad=15)
            generated_code = f"sns.boxplot(data=df, y='{num_col}')"
            observation = f"Box plot highlighting median, IQR range, and extreme values for '{num_col}'."

    elif "heatmap" in chart_type_lower or "corr" in chart_type_lower:
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True, linewidths=0.5)
            ax.set_title(title or "Correlation Heatmap Matrix", fontsize=14, color="#f8fafc", pad=15)
            generated_code = f"sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')"
            observation = f"Correlation matrix displaying pairwise relationships across {len(numeric_cols)} numerical features."
        else:
            sns.histplot(data=df, x=df.columns[0], ax=ax)
            observation = "Insufficient numerical variables for correlation heatmap."

    else:
        # Default bar plot fallback
        col = x_col or df.columns[0]
        sns.histplot(data=df, x=col, ax=ax, color="#38bdf8")
        ax.set_title(title or f"Overview Chart of {col}", fontsize=14, color="#f8fafc", pad=15)
        generated_code = f"sns.histplot(data=df, x='{col}')"
        observation = f"Distribution plot generated for feature '{col}'."

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    # Save to disk
    filename = f"chart_{uuid.uuid4().hex[:10]}.png"
    save_path = os.path.join(settings.GENERATED_PATH, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='#181b20')
    plt.close(fig)

    base64_img = encode_image_to_base64(save_path)

    return {
        "chart_type": chart_type,
        "title": title or f"{chart_type.title()} Chart",
        "image_path": save_path,
        "filename": filename,
        "base64_image": base64_img,
        "code": generated_code,
        "observations": observation
    }
