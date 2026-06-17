"""
DataSense utilities package.
Contains scanner, visualizer, LLM integration, and chat utilities.
"""

from .scanner import scan_csv, suggest_target, suggest_ml_models
from .visualizer import (
    plot_distributions,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_categorical_counts,
    plot_scatter_matrix,
)
from .llm import generate_summary, chat_with_data
from .chat import execute_pandas_code, result_to_chart

__all__ = [
    "scan_csv",
    "suggest_target",
    "suggest_ml_models",
    "plot_distributions",
    "plot_correlation_heatmap",
    "plot_missing_values",
    "plot_categorical_counts",
    "plot_scatter_matrix",
    "generate_summary",
    "chat_with_data",
    "execute_pandas_code",
    "result_to_chart",
]
