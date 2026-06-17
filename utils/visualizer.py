"""
Plotly visualization functions for DataSense.
Generates various charts for data exploration and analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from typing import Optional


def plot_distributions(
    df: pd.DataFrame, numeric_cols: list, max_cols: int = 6
) -> list:
    """
    Create histogram + KDE overlay charts for numeric columns.
    
    Args:
        df: pandas DataFrame
        numeric_cols: list of numeric column names
        max_cols: maximum number of columns to plot
        
    Returns:
        List of Plotly figures
    """
    figures = []
    cols_to_plot = numeric_cols[:max_cols]
    
    for col in cols_to_plot:
        data = df[col].dropna()
        
        if len(data) == 0:
            continue
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                name="Count",
                marker=dict(color="#6366f1", opacity=0.7),
                nbinsx=30,
            )
        )
        
        # KDE overlay
        try:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            kde_values = kde(x_range)
            
            # Scale KDE to match histogram
            bin_width = (data.max() - data.min()) / 30
            kde_scaled = kde_values * len(data) * bin_width
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_scaled,
                    mode="lines",
                    name="KDE",
                    line=dict(color="#f59e0b", width=2),
                )
            )
        except Exception:
            pass
        
        # Mean line
        mean_val = data.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}",
        )
        
        fig.update_layout(
            title=f"Distribution: {col}",
            xaxis_title=col,
            yaxis_title="Frequency",
            hovermode="x unified",
            template="plotly_dark",
            height=400,
        )
        
        figures.append(fig)
    
    return figures


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> Optional[go.Figure]:
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df: pandas DataFrame
        numeric_cols: list of numeric column names
        
    Returns:
        Plotly figure or None if fewer than 2 numeric columns
    """
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{y} - %{x}: %{z:.2f}<extra></extra>",
        )
    )
    
    fig.update_layout(
        title="Correlation Matrix",
        template="plotly_dark",
        height=500,
        width=600,
    )
    
    return fig


def plot_missing_values(df: pd.DataFrame, scan: dict) -> Optional[go.Figure]:
    """
    Create a horizontal bar chart showing percentage of missing values.
    
    Args:
        df: pandas DataFrame
        scan: scan dict from scan_csv()
        
    Returns:
        Plotly figure or None if no missing values
    """
    if not scan["missing_pct"]:
        return None
    
    missing_data = scan["missing_pct"]
    cols = list(missing_data.keys())
    pcts = list(missing_data.values())
    
    # Color coding
    colors = []
    for pct in pcts:
        if pct > 30:
            colors.append("red")
        elif pct > 10:
            colors.append("orange")
        else:
            colors.append("gold")
    
    fig = go.Figure(
        data=go.Bar(
            y=cols,
            x=pcts,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{p}%" for p in pcts],
            textposition="auto",
        )
    )
    
    fig.update_layout(
        title="Missing Values by Column",
        xaxis_title="% Missing",
        yaxis_title="Column",
        template="plotly_dark",
        height=400 if len(cols) <= 5 else 300 + len(cols) * 30,
    )
    
    return fig


def plot_categorical_counts(
    df: pd.DataFrame, categorical_cols: list, max_cols: int = 4
) -> list:
    """
    Create bar charts for top values in categorical columns.
    
    Args:
        df: pandas DataFrame
        categorical_cols: list of categorical column names
        max_cols: maximum number of columns to plot
        
    Returns:
        List of Plotly figures
    """
    figures = []
    cols_to_plot = categorical_cols[:max_cols]
    
    for col in cols_to_plot:
        value_counts = df[col].value_counts().head(10)
        
        fig = go.Figure(
            data=go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                marker=dict(color="#10b981"),
            )
        )
        
        fig.update_layout(
            title=f"Top Values: {col}",
            xaxis_title=col,
            yaxis_title="Count",
            template="plotly_dark",
            height=400,
            hovermode="x unified",
        )
        
        figures.append(fig)
    
    return figures


def plot_scatter_matrix(
    df: pd.DataFrame, numeric_cols: list, target_col: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Create a scatter matrix for numeric columns.
    
    Args:
        df: pandas DataFrame
        numeric_cols: list of numeric column names
        target_col: optional target column for coloring (must be categorical)
        
    Returns:
        Plotly figure or None if fewer than 2 numeric columns
    """
    if len(numeric_cols) < 2:
        return None
    
    cols_subset = numeric_cols[:5]  # Limit to 5 for performance
    
    if target_col and target_col in df.columns:
        fig = px.scatter_matrix(
            df[cols_subset + [target_col]],
            dimensions=cols_subset,
            color=target_col,
            title="Scatter Matrix",
            template="plotly_dark",
            height=800,
        )
    else:
        fig = px.scatter_matrix(
            df[cols_subset],
            title="Scatter Matrix",
            template="plotly_dark",
            height=800,
        )
    
    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
    
    return fig
