"""
Utilities for executing pandas code and generating charts from results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Any, Optional


def execute_pandas_code(code: str, df: pd.DataFrame) -> tuple[Any, Optional[str]]:
    """
    Safely execute pandas code and return the result.
    
    Args:
        code: Python code string (must assign result to df_result)
        df: pandas DataFrame available as 'df'
        
    Returns:
        tuple of (result, error_message)
        result is None if error occurred
    """
    try:
        local_vars = {"df": df, "pd": pd, "np": np}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        
        if "df_result" in local_vars:
            return local_vars["df_result"], None
        else:
            return None, "Code did not assign result to df_result"
    except Exception as e:
        return None, f"Error executing code: {str(e)}"


def result_to_chart(
    result: Any,
    chart_type: Optional[str],
    chart_x: Optional[str],
    chart_y: Optional[str]
) -> Optional[go.Figure]:
    """
    Convert a pandas result to a Plotly chart.
    
    Args:
        result: pandas Series or DataFrame
        chart_type: "bar", "line", "histogram", "scatter", or None
        chart_x: x-axis column name
        chart_y: y-axis column name
        
    Returns:
        Plotly figure or None
    """
    if result is None or chart_type is None or chart_type == "none":
        return None
    
    try:
        if chart_type == "bar":
            if isinstance(result, pd.Series):
                fig = go.Figure(
                    data=go.Bar(x=result.index, y=result.values, marker_color="#6366f1")
                )
                fig.update_layout(
                    title="Bar Chart",
                    xaxis_title=result.index.name or "Category",
                    yaxis_title="Value",
                    template="plotly_dark",
                )
            elif isinstance(result, pd.DataFrame):
                if chart_x and chart_y:
                    fig = px.bar(result, x=chart_x, y=chart_y)
                    fig.update_layout(template="plotly_dark")
                else:
                    return None
            else:
                return None
        
        elif chart_type == "line":
            if isinstance(result, pd.Series):
                fig = go.Figure(
                    data=go.Scatter(x=result.index, y=result.values, mode="lines+markers")
                )
                fig.update_layout(
                    title="Line Chart",
                    xaxis_title=result.index.name or "X",
                    yaxis_title="Value",
                    template="plotly_dark",
                )
            elif isinstance(result, pd.DataFrame):
                if chart_x and chart_y:
                    fig = px.line(result, x=chart_x, y=chart_y)
                    fig.update_layout(template="plotly_dark")
                else:
                    return None
            else:
                return None
        
        elif chart_type == "histogram":
            if isinstance(result, pd.Series):
                fig = go.Figure(data=go.Histogram(x=result, marker_color="#6366f1"))
                fig.update_layout(
                    title="Histogram",
                    xaxis_title=result.name or "Value",
                    yaxis_title="Frequency",
                    template="plotly_dark",
                )
            else:
                return None
        
        elif chart_type == "scatter":
            if isinstance(result, pd.DataFrame):
                if chart_x and chart_y:
                    fig = px.scatter(result, x=chart_x, y=chart_y)
                    fig.update_layout(template="plotly_dark")
                else:
                    return None
            else:
                return None
        
        else:
            return None
        
        return fig
    
    except Exception:
        return None
