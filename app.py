"""
DataSense - CSV & Dataset Analyzer
Main Streamlit application with multi-stage workflow.
"""

import streamlit as st
import pandas as pd
import json
from utils.scanner import scan_csv, suggest_target, suggest_ml_models
from utils.visualizer import (
    plot_distributions,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_categorical_counts,
    plot_scatter_matrix,
)
from utils.llm import generate_summary, chat_with_data
from utils.chat import execute_pandas_code, result_to_chart


# Page configuration
st.set_page_config(
    page_title="DataSense",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0f0f13;
    --bg-card: #1a1a24;
    --accent-primary: #6366f1;
    --accent-success: #10b981;
    --text-primary: #e2e2e8;
    --border-radius: 12px;
}

* {
    font-family: 'Inter', sans-serif;
}

body {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-primary);
}

.stMetric, .stInfo, .stWarning, .stError {
    background-color: var(--bg-card) !important;
    border-radius: var(--border-radius);
    padding: 1rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
}

[data-testid="stMetricValue"] {
    color: var(--accent-primary);
    font-size: 2.5rem;
}

.upload-area {
    border: 2px dashed var(--accent-primary) !important;
    border-radius: var(--border-radius);
    padding: 2rem;
}

.mode-card {
    background-color: var(--bg-card);
    border-radius: var(--border-radius);
    padding: 2rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
    border-left: 4px solid var(--accent-primary);
}

.quality-score-excellent {
    color: #10b981;
}

.quality-score-good {
    color: #f59e0b;
}

.quality-score-poor {
    color: #ef4444;
}

.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "scan" not in st.session_state:
    st.session_state.scan = None
if "mode" not in st.session_state:
    st.session_state.mode = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stage" not in st.session_state:
    st.session_state.stage = "upload"


# Helper functions
def reset_app():
    """Reset app to initial state."""
    st.session_state.df = None
    st.session_state.scan = None
    st.session_state.mode = None
    st.session_state.summary = None
    st.session_state.chat_history = []
    st.session_state.stage = "upload"


def show_stage_upload():
    """Stage 1: File upload screen."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; font-size: 3.5rem; margin: 0;'>📊 DataSense</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; font-size: 1.2rem; color: #999; margin-top: 0.5rem;'>Upload any CSV. Understand it instantly.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file to get started"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("📂 Reading file..."):
                    df = pd.read_csv(uploaded_file)
                
                if len(df) == 0:
                    st.error("The CSV file is empty.")
                    return
                
                # Handle large files
                if len(df) > 10000:
                    st.warning(f"⚠️ File has {len(df):,} rows. Sampling 10,000 rows for analysis.")
                    df = df.sample(10000, random_state=42)
                
                # Check file size
                file_size_mb = uploaded_file.size / (1024 ** 2)
                if file_size_mb > 50:
                    st.warning("⚠️ Large file detected. Analysis may be slower.")
                
                with st.spinner("🔍 Scanning your data..."):
                    scan = scan_csv(df)
                
                st.session_state.df = df
                st.session_state.scan = scan
                st.session_state.stage = "mode_select"
                
                st.success("✅ File uploaded successfully!")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")


def show_stage_mode_select():
    """Stage 2: Mode selection screen."""
    st.markdown("<br>", unsafe_allow_html=True)
    # Show scan summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{st.session_state.scan['row_count']:,}")
    with col2:
        st.metric("Columns", st.session_state.scan['col_count'])
    with col3:
        score = st.session_state.scan['data_quality_score']
        if score >= 80:
            st.metric("Quality Score", f"{score:.0f}", delta="Excellent")
        elif score >= 60:
            st.metric("Quality Score", f"{score:.0f}", delta="Good")
        else:
            st.metric("Quality Score", f"{score:.0f}", delta="Needs work")
    with col4:
        st.metric("Memory Usage", f"{st.session_state.scan['memory_mb']} MB")
    
    st.info(st.session_state.scan['one_line'])
    
    st.markdown("---")
    st.markdown("## Choose Your Analysis Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("### ⚡ Quick Insights")
            st.markdown(
                "Plain English summary, clean charts, simple Q&A. No jargon."
            )
            if st.button("Get Quick Insights", key="quick_mode", use_container_width=True):
                st.session_state.mode = "quick"
                st.session_state.stage = "dashboard"
                with st.spinner("✨ Generating summary..."):
                    st.session_state.summary = generate_summary(
                        st.session_state.scan,
                        st.session_state.df,
                        "quick"
                    )
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.markdown("### 🧠 ML Analysis")
            st.markdown(
                "Data quality score, correlation analysis, model suggestions, technical chat."
            )
            if st.button("Start ML Analysis", key="ml_mode", use_container_width=True):
                st.session_state.mode = "ml"
                st.session_state.stage = "dashboard"
                with st.spinner("✨ Generating summary..."):
                    st.session_state.summary = generate_summary(
                        st.session_state.scan,
                        st.session_state.df,
                        "ml"
                    )
                st.rerun()


def show_stage_dashboard():
    """Stage 3: Main dashboard with tabs."""
    # Navigation buttons
    col_back1, col_back2, col_spacer = st.columns([1, 1, 4])
    with col_back1:
        if st.button("← Change Mode"):
            st.session_state.mode = None
            st.session_state.summary = None
            st.session_state.chat_history = []
            st.session_state.stage = "mode_select"
            st.rerun()
    with col_back2:
        if st.button("⬆ New File"):
            reset_app()
            st.rerun()

    st.divider()
    
    # Main tabs
    if st.session_state.mode == "quick":
        tabs = st.tabs(["📋 Summary", "💬 Chat with Data"])
    else:
        tabs = st.tabs(["📋 Summary", "📊 Charts", "💬 Chat with Data", "🧠 ML Insights"])
    # Tab 1: Summary
    with tabs[0]:
        st.subheader("Dataset Summary")
        st.markdown(st.session_state.summary)
    
    # Tab 2: Charts
    if st.session_state.mode == "ml":
        with tabs[1]:
            st.subheader("Data Visualizations")
        
            col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.scan['numeric_cols']:
                st.markdown("#### Distributions")
                dist_figs = plot_distributions(
                    st.session_state.df,
                    st.session_state.scan['numeric_cols'],
                    max_cols=3
                )
                for fig in dist_figs:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.session_state.scan['categorical_cols']:
                st.markdown("#### Top Values")
                cat_figs = plot_categorical_counts(
                    st.session_state.df,
                    st.session_state.scan['categorical_cols'],
                    max_cols=3
                )
                for fig in cat_figs:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Scatter matrix
        if len(st.session_state.scan['numeric_cols']) >= 2:
            st.markdown("#### Scatter Matrix")
            scatter_fig = plot_scatter_matrix(
                st.session_state.df,
                st.session_state.scan['numeric_cols']
            )
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Tab 3: Chat with Data
    chat_tab_index = 1 if st.session_state.mode == "quick" else 2
    with tabs[chat_tab_index]:
        if st.session_state.mode == "quick":
            st.markdown("**💬 Ask questions in plain English — no coding needed!**")
        else:
            st.markdown("**💬 Ask technical questions — I'll write pandas code to answer them.**")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("code"):
                    with st.expander("View Code"):
                        st.code(msg["code"], language="python")
                if msg["role"] == "assistant" and msg.get("figure"):
                    st.plotly_chart(msg["figure"], use_container_width=True)
        
        # Chat input
        user_input = st.chat_input("Ask anything about your data...")
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
            })
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get response
            with st.spinner("🤔 Thinking..."):
                result = chat_with_data(
                    user_input,
                    st.session_state.df,
                    st.session_state.scan,
                    [{"role": m["role"], "content": m["content"]}
                     for m in st.session_state.chat_history[:-1]],
                    st.session_state.mode
                )
            
            # Process response
            with st.chat_message("assistant"):
                st.markdown(result["answer"])
                
                # Execute code if available
                figure = None
                if result.get("code"):
                    with st.expander("View Code"):
                        st.code(result["code"], language="python")
                    
                    exec_result, error = execute_pandas_code(
                        result["code"],
                        st.session_state.df
                    )
                    
                    if error:
                        st.error(f"Code execution error: {error}")
                    else:
                        # Try to generate chart
                        figure = result_to_chart(
                            exec_result,
                            result.get("chart_type"),
                            result.get("chart_x"),
                            result.get("chart_y")
                        )
                        if figure:
                            st.plotly_chart(figure, use_container_width=True)
                
                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "code": result.get("code"),
                    "figure": figure,
                })
            
            st.rerun()
    
    # Tab 4: ML Insights (ML mode only)
    if st.session_state.mode == "ml" and len(tabs) > 3:
        with tabs[3]:
            st.subheader("Machine Learning Insights")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Data quality
                score = st.session_state.scan['data_quality_score']
                color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                st.metric(
                    "Data Quality Score",
                    f"{score:.0f}",
                    delta=f"Quality: {'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Needs Improvement'}"
                )
                
                # Target suggestion
                target = suggest_target(st.session_state.df, st.session_state.scan)
                if target:
                    target_dtype = st.session_state.scan['dtypes'].get(target, "unknown")
                    st.success(f"**Suggested Target:** `{target}` ({target_dtype})")
                else:
                    st.info("No obvious target column detected. Consider unsupervised learning.")
            
            with col2:
                st.markdown("#### Column Types")
                st.write(f"- Numeric: {len(st.session_state.scan['numeric_cols'])}")
                st.write(f"- Categorical: {len(st.session_state.scan['categorical_cols'])}")
            
            st.divider()
            
            # Model recommendations
            st.subheader("Recommended Models")
            models = suggest_ml_models(
                st.session_state.df,
                st.session_state.scan,
                suggest_target(st.session_state.df, st.session_state.scan)
            )
            
            if models:
                cols = st.columns(len(models))
                for idx, model in enumerate(models):
                    with cols[idx]:
                    with st.container(border=True):
                        st.markdown(f"### {model['model']}")
                        st.markdown(f"*{model['reason']}*")
                        st.code(f"from {model['sklearn_class'].rsplit('.', 1)[0]} import {model['sklearn_class'].rsplit('.', 1)[1]}")
            
            st.divider()
            
            # Feature analysis
            st.subheader("Feature Engineering Hints")
            numeric_cols = st.session_state.scan['numeric_cols']
            if numeric_cols:
                variances = st.session_state.df[numeric_cols].var()
                top_features = variances.nlargest(5)
                st.markdown("**Top Features by Variance:**")
                for col, var in top_features.items():
                    st.write(f"- `{col}`: {var:.2f}")
            
            # Correlation insights
            if len(numeric_cols) >= 2:
                st.markdown("**Top Correlations:**")
                corr_matrix = st.session_state.df[numeric_cols].corr()
                
                # Get top correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            "col1": corr_matrix.columns[i],
                            "col2": corr_matrix.columns[j],
                            "corr": abs(corr_matrix.iloc[i, j])
                        })
                
                corr_pairs.sort(key=lambda x: x["corr"], reverse=True)
                for pair in corr_pairs[:3]:
                    st.write(
                        f"- `{pair['col1']}` ↔ `{pair['col2']}`: {corr_matrix.loc[pair['col1'], pair['col2']]:+.2f}"
                    )


# Main app flow
if st.session_state.stage == "upload":
    show_stage_upload()
elif st.session_state.stage == "mode_select":
    show_stage_mode_select()
elif st.session_state.stage == "dashboard":
    show_stage_dashboard()
