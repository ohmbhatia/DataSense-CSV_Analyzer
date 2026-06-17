"""
LLM integration for DataSense with support for both Groq and Anthropic Claude.
Handles summary generation and chat interactions.
Provider selection via LLM_PROVIDER environment variable: 'groq' or 'anthropic'.
"""

import json
import os
from typing import Optional
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Import providers based on availability
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def get_provider_and_key():
    """Get LLM provider and API key from environment or Streamlit secrets."""
    try:
        import streamlit as st
        provider = st.secrets.get("LLM_PROVIDER") or os.environ.get("LLM_PROVIDER", "groq")
        
        if provider == "groq":
            api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
        else:  # anthropic
            api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    except Exception:
        # Fallback for non-Streamlit environments
        provider = os.environ.get("LLM_PROVIDER", "groq")
        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
    return provider, api_key



def generate_summary(scan: dict, df: pd.DataFrame, mode: str) -> str:
    """
    Generate a natural language summary of the dataset using Claude or Groq.
    
    Args:
        scan: scan dict from scan_csv()
        df: pandas DataFrame
        mode: "quick" or "ml"
        
    Returns:
        Summary string from LLM
    """
    provider, api_key = get_provider_and_key()
    
    if not api_key:
        return f"API key not found. Please set {provider.upper()}_API_KEY environment variable or Streamlit secret."
    
    # Prepare data context
    first_rows = df.head(5).to_markdown(index=False)
    scan_json = json.dumps(scan, indent=2, default=str)
    
    if mode == "quick":
        system_prompt = """You are a friendly data analyst. Write a plain English summary of this dataset for a non-technical user.
Use simple language, no jargon. Structure your response as:
1. What this data appears to be about (2 sentences)
2. Key highlights (3–4 bullet points with plain observations)
3. One interesting thing to investigate further
Keep it under 200 words. Do not mention column names directly — describe them naturally."""
    else:  # ml mode
        system_prompt = """You are a senior ML engineer reviewing a dataset. Write a technical data quality and ML-readiness summary.
Structure your response as:
1. Dataset Overview (size, types, quality score assessment)
2. Data Quality Issues (missing values, duplicates, suspicious patterns)
3. Feature Engineering Opportunities (based on column types and cardinalities)
4. ML Readiness Assessment (what tasks this data is suitable for)
5. Recommended Next Steps (3 concrete actions before modeling)
Be specific, reference column names, be direct about issues."""
    
    user_message = f"""Please analyze this dataset:

**Dataset Statistics:**
{scan_json}

**First 5 rows:**
{first_rows}

Provide your analysis based on the structure above."""
    
    if provider == "groq":
        if not GROQ_AVAILABLE:
            return "Groq SDK not installed. Run: pip install groq"
        
        client = Groq(api_key=api_key)
        # Groq requires system prompt as first message with role "system"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=1024,
            messages=messages,
        )
        return response.choices[0].message.content
    else:  # anthropic
        if not ANTHROPIC_AVAILABLE:
            return "Anthropic SDK not installed. Run: pip install anthropic"
        
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


def chat_with_data(
    question: str,
    df: pd.DataFrame,
    scan: dict,
    history: list[dict],
    mode: str
) -> dict:
    """
    Chat with the dataset, generating pandas code to answer questions.
    Uses either Groq or Anthropic Claude based on LLM_PROVIDER setting.
    
    Args:
        question: user question
        df: pandas DataFrame
        scan: scan dict from scan_csv()
        history: conversation history
        mode: "quick" or "ml"
        
    Returns:
        dict with keys: answer, code, error, chart_type, chart_x, chart_y
    """
    provider, api_key = get_provider_and_key()
    
    if not api_key:
        return {
            "answer": f"API key not found. Please set {provider.upper()}_API_KEY environment variable or Streamlit secret.",
            "code": None,
            "error": "API key missing",
            "chart_type": None,
            "chart_x": None,
            "chart_y": None,
        }
    
    # Build system prompt
    if mode == "quick":
        mode_instruction = """The user has uploaded a CSV dataset. Answer their questions in plain English.
You can suggest what analysis would be helpful, but don't assume they know pandas/coding.
If the answer would benefit from code, you can suggest it, but keep language simple."""
    else:
        mode_instruction = """You are a data analysis assistant. The user has uploaded a CSV dataset.
You have access to a pandas DataFrame called `df`.
When the user asks a question:
1. Think about what pandas/numpy code would answer it
2. Return your response in this EXACT JSON format:
{
  "answer": "Plain English answer to the question",
  "code": "df_result = df.groupby('col').mean()  # valid pandas code",
  "chart_type": "bar|line|histogram|scatter|none",
  "chart_x": "column name or null",
  "chart_y": "column name or null"
}
Only return JSON. If the question can't be answered, set code to null."""
    
    system_prompt = f"""{mode_instruction}

Column info: {json.dumps(scan['dtypes'])}
Shape: {scan['row_count']} rows × {scan['col_count']} columns
Data Quality Score: {scan['data_quality_score']}
"""
    
    # Build messages
    messages = history.copy()
    messages.append({"role": "user", "content": question})
    
    try:
        if provider == "groq":
            if not GROQ_AVAILABLE:
                return {
                    "answer": "Groq SDK not installed. Run: pip install groq",
                    "code": None,
                    "error": "Groq SDK missing",
                    "chart_type": None,
                    "chart_x": None,
                    "chart_y": None,
                }
            
            client = Groq(api_key=api_key)
            # Groq requires system prompt as first message with role "system"
            groq_messages = [{"role": "system", "content": system_prompt}] + messages
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=1024,
                messages=groq_messages,
            )
            response_text = response.choices[0].message.content
        else:  # anthropic
            if not ANTHROPIC_AVAILABLE:
                return {
                    "answer": "Anthropic SDK not installed. Run: pip install anthropic",
                    "code": None,
                    "error": "Anthropic SDK missing",
                    "chart_type": None,
                    "chart_x": None,
                    "chart_y": None,
                }
            
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
            response_text = response.content[0].text
        
        if mode == "ml":
            # Try to parse as JSON
            try:
                result = json.loads(response_text)
                return {
                    "answer": result.get("answer", ""),
                    "code": result.get("code"),
                    "error": None,
                    "chart_type": result.get("chart_type"),
                    "chart_x": result.get("chart_x"),
                    "chart_y": result.get("chart_y"),
                }
            except json.JSONDecodeError:
                return {
                    "answer": response_text,
                    "code": None,
                    "error": "Could not parse response as JSON",
                    "chart_type": None,
                    "chart_x": None,
                    "chart_y": None,
                }
        else:
            # Quick mode - just return text
            return {
                "answer": response_text,
                "code": None,
                "error": None,
                "chart_type": None,
                "chart_x": None,
                "chart_y": None,
            }
    
    except Exception as e:
        return {
            "answer": f"Error communicating with {provider}: {str(e)}",
            "code": None,
            "error": str(e),
            "chart_type": None,
            "chart_x": None,
            "chart_y": None,
        }
