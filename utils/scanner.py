"""
CSV scanning and analysis utilities for DataSense.
Provides functions to analyze datasets and suggest targets/models.
"""

import pandas as pd
import numpy as np
from typing import Optional


def scan_csv(df: pd.DataFrame) -> dict:
    """
    Scan a DataFrame and return comprehensive summary statistics.
    
    Args:
        df: pandas DataFrame to scan
        
    Returns:
        dict with keys: row_count, col_count, columns, dtypes, missing, missing_pct,
                       duplicates, numeric_cols, categorical_cols, cardinality,
                       memory_mb, one_line, data_quality_score
    """
    row_count = len(df)
    col_count = len(df.columns)
    columns = list(df.columns)
    
    # Convert dtypes to readable strings
    dtypes = {}
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            dtypes[col] = "numeric"
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            dtypes[col] = "datetime"
        elif pd.api.types.is_bool_dtype(dtype):
            dtypes[col] = "boolean"
        else:
            dtypes[col] = "categorical"
            categorical_cols.append(col)
    
    # Missing values
    missing = {}
    missing_pct = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing[col] = int(missing_count)
            missing_pct[col] = round(100 * missing_count / row_count, 2)
    
    # Duplicates
    duplicates = int(df.duplicated().sum())
    
    # Cardinality for categorical columns
    cardinality = {}
    for col in categorical_cols:
        cardinality[col] = int(df[col].nunique())
    
    # Memory usage in MB
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)
    
    # One-line description
    numeric_desc = f"{len(numeric_cols)} numeric"
    categorical_desc = f"{len(categorical_cols)} categorical"
    missing_avg = round(np.mean(list(missing_pct.values())) if missing_pct else 0, 1)
    one_line = f"{row_count} rows × {col_count} columns | {numeric_desc}, {categorical_desc} | {missing_avg}% missing values"
    
    # Data quality score (0-100)
    quality_score = 100.0
    
    # Subtract for high missing values
    for col, pct in missing_pct.items():
        if pct > 30:
            quality_score -= 10
        elif pct > 5:
            quality_score -= 5
    
    # Subtract for duplicates
    if row_count > 0 and duplicates > (0.01 * row_count):
        quality_score -= 5
    
    # Subtract for constant columns
    for col in cardinality:
        if cardinality[col] == 1:
            quality_score -= 3
    
    data_quality_score = max(0, min(100, quality_score))
    
    return {
        "row_count": row_count,
        "col_count": col_count,
        "columns": columns,
        "dtypes": dtypes,
        "missing": missing,
        "missing_pct": missing_pct,
        "duplicates": duplicates,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cardinality": cardinality,
        "memory_mb": memory_mb,
        "one_line": one_line,
        "data_quality_score": round(data_quality_score, 1),
    }


def suggest_target(df: pd.DataFrame, scan: dict) -> Optional[str]:
    """
    Suggest a target column for ML tasks based on column names.
    
    Args:
        df: pandas DataFrame
        scan: scan dict from scan_csv()
        
    Returns:
        Column name or None if no suitable target found
    """
    target_keywords = [
        "target",
        "label",
        "class",
        "output",
        "y",
        "result",
        "outcome",
        "survived",
        "churn",
        "fraud",
        "price",
        "sales",
        "revenue",
    ]
    
    # Exact match first
    for keyword in target_keywords:
        for col in df.columns:
            if col.lower() == keyword:
                return col
    
    # Partial match
    for keyword in target_keywords:
        for col in df.columns:
            if keyword in col.lower():
                return col
    
    return None


def suggest_ml_models(
    df: pd.DataFrame, scan: dict, target_col: Optional[str] = None
) -> list[dict]:
    """
    Suggest ML models based on dataset characteristics.
    
    Args:
        df: pandas DataFrame
        scan: scan dict from scan_csv()
        target_col: target column name if known
        
    Returns:
        List of dicts with keys: model, reason, sklearn_class
    """
    suggestions = []
    
    if target_col and target_col in df.columns:
        target_dtype = scan["dtypes"].get(target_col)
        
        if target_dtype == "numeric":
            # Regression
            suggestions = [
                {
                    "model": "Linear Regression",
                    "reason": "Fast baseline for numeric predictions",
                    "sklearn_class": "sklearn.linear_model.LinearRegression",
                },
                {
                    "model": "Random Forest Regressor",
                    "reason": "Handles non-linear patterns and feature interactions",
                    "sklearn_class": "sklearn.ensemble.RandomForestRegressor",
                },
                {
                    "model": "XGBoost Regressor",
                    "reason": "State-of-the-art gradient boosting for regression",
                    "sklearn_class": "xgboost.XGBRegressor",
                },
            ]
        else:
            # Classification
            unique_values = df[target_col].nunique()
            if unique_values <= 10:
                suggestions = [
                    {
                        "model": "Logistic Regression",
                        "reason": "Interpretable baseline for classification",
                        "sklearn_class": "sklearn.linear_model.LogisticRegression",
                    },
                    {
                        "model": "Random Forest Classifier",
                        "reason": "Robust to feature scaling and handles interactions",
                        "sklearn_class": "sklearn.ensemble.RandomForestClassifier",
                    },
                    {
                        "model": "XGBoost Classifier",
                        "reason": "Industry-standard for classification tasks",
                        "sklearn_class": "xgboost.XGBClassifier",
                    },
                ]
    else:
        # Unsupervised learning
        suggestions = [
            {
                "model": "KMeans Clustering",
                "reason": "Partition data into similar groups",
                "sklearn_class": "sklearn.cluster.KMeans",
            },
            {
                "model": "DBSCAN",
                "reason": "Density-based clustering, discovers arbitrary shapes",
                "sklearn_class": "sklearn.cluster.DBSCAN",
            },
            {
                "model": "PCA (Dimensionality Reduction)",
                "reason": "Reduce dimensions while preserving variance",
                "sklearn_class": "sklearn.decomposition.PCA",
            },
        ]
    
    return suggestions
