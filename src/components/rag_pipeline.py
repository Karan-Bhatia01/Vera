"""
Simplified Chatbot - Database Direct Integration (No FAISS/RAG).

Simple chatbot that:
1. Retrieves dataset context from MongoDB
2. Sends query + context to LLM
3. Returns answer about that specific dataset only
"""

import os
import openai
from typing import Optional
from src.logger import logging

OXLO_BASE_URL = "https://api.oxlo.ai/v1"
LLM_MODEL = "deepseek-r1-8b"


def _get_client():
    """Create OpenAI client pointing to Oxlo."""
    return openai.OpenAI(
        base_url=OXLO_BASE_URL,
        api_key=os.environ.get("OXLO_API_KEY", ""),
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET DATASET CONTEXT FROM MONGODB
# ──────────────────────────────────────────────────────────────────────────────


def get_dataset_context(filename: str) -> dict:
    """
    Retrieve dataset context from MongoDB for a specific dataset.
    Returns analysis, AI insights, and metadata.
    """
    try:
        from src.components.mongo_storage import get_dataset_insights
        from src.utils import load_dataframe_from_mongo
        
        # Load the dataframe for basic stats
        df = load_dataframe_from_mongo(filename)
        
        context = {
            "filename": filename,
            "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
        }
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
        
        context["numeric_columns"] = numeric_cols
        context["categorical_columns"] = categorical_cols
        
        # Add statistics for numeric columns
        stats = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) > 0:
                stats[col] = {
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "median": float(s.median()),
                }
        context["numeric_stats"] = stats
        
        # Add value distributions for categorical columns
        distributions = {}
        for col in categorical_cols:
            vc = df[col].value_counts().head(5).to_dict()
            distributions[col] = vc
        context["categorical_distributions"] = distributions
        
        # Get AI insights from MongoDB
        doc = get_dataset_insights(filename)
        if doc:
            ai_insights = doc.get("ai_insights", {})
            context["ai_summary"] = ai_insights.get("summary", "")
            context["quality_flags"] = ai_insights.get("quality_flags", [])
            context["column_insights"] = ai_insights.get("column_insights", [])
        
        logging.info(f"✅ Retrieved context for dataset: {filename}")
        return context
        
    except Exception as e:
        logging.error(f"❌ Failed to get dataset context: {e}")
        return {
            "filename": filename,
            "error": str(e),
        }


def format_context_for_prompt(context: dict) -> str:
    """Format dataset context as readable text for the LLM prompt."""
    if "error" in context:
        return f"Could not retrieve dataset: {context['error']}"
    
    lines = []
    lines.append(f"📊 Dataset: {context.get('filename', 'Unknown')}")
    lines.append(f"Shape: {context.get('shape', 'Unknown')}")
    
    if context.get("columns"):
        cols = context["columns"]
        cols_str = ", ".join(str(c) for c in cols[:10])
        if len(cols) > 10:
            cols_str += f"... (+{len(cols) - 10} more)"
        lines.append(f"Columns ({len(cols)}): {cols_str}")
    
    if context.get("missing_values") is not None:
        lines.append(f"Missing values: {context['missing_values']}")
    
    if context.get("duplicates") is not None:
        lines.append(f"Duplicate rows: {context['duplicates']}")
    
    if context.get("numeric_columns"):
        lines.append(f"Numeric columns: {', '.join(context['numeric_columns'])}")
    
    if context.get("categorical_columns"):
        lines.append(f"Categorical columns: {', '.join(context['categorical_columns'])}")
    
    # Add key statistics
    if context.get("numeric_stats"):
        lines.append("\nKey Statistics:")
        for col, stats in list(context["numeric_stats"].items())[:3]:
            lines.append(f"  {col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
    
    # Add AI summary
    if context.get("ai_summary"):
        lines.append(f"\nAI Analysis: {context['ai_summary'][:200]}...")
    
    # Add quality issues
    if context.get("quality_flags"):
        lines.append("\nData Quality Issues:")
        for flag in context["quality_flags"][:3]:
            lines.append(f"  - [{flag.get('severity', 'INFO')}] {flag.get('column', '')}: {flag.get('issue', '')}")
    
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CHAT FUNCTION - THE MAIN INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Vera, a helpful data science assistant.

Answer questions about the dataset provided in the context.
Be accurate, concise, and reference actual values when possible.
If asked about something not in the dataset, politely redirect to the dataset.
"""


def chat(
    query: str,
    filename: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> dict:
    """
    Chat with the dataset.
    
    Args:
        query: User's question
        filename: Which dataset to ask about (required for dataset questions)
        history: Previous messages for context
    
    Returns:
        Dict with: answer, dataset, context_available
    """
    
    context_str = ""
    context_available = False
    
    # If a dataset is specified, get its context
    if filename:
        try:
            context_dict = get_dataset_context(filename)
            if "error" not in context_dict:
                context_str = format_context_for_prompt(context_dict)
                context_available = True
                logging.info(f"Using context for dataset: {filename}")
            else:
                logging.warning(f"No context available for {filename}")
        except Exception as e:
            logging.error(f"Error getting context: {e}")
    
    # Build the message
    if context_str:
        user_message = f"Dataset Information:\n{context_str}\n\nQuestion: {query}"
    else:
        user_message = f"Question: {query}"
    
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history (keep last 6 messages = 3 turns)
    if history:
        messages.extend(history[-6:])
    
    # Add current question
    messages.append({"role": "user", "content": user_message})
    
    # Call LLM
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1024,
            temperature=0.2,
            messages=messages,
            timeout=60.0,
        )
        
        answer = response.choices[0].message.content.strip()
        logging.info(f"✅ Chat response generated")
        
    except Exception as e:
        logging.error(f"❌ Chat failed: {e}")
        answer = f"Error: {str(e)[:80]}. Please try again."
    
    return {
        "answer": answer,
        "dataset": filename,
        "context_available": context_available,
    }


# ──────────────────────────────────────────────────────────────────────────────
# BACKWARD COMPATIBILITY (for existing code that might call these)
# ──────────────────────────────────────────────────────────────────────────────


def build_vector_store(filename: str) -> int:
    """Deprecated - kept for backward compatibility."""
    logging.info(f"Note: build_vector_store called (deprecated, FAISS disabled)")
    return 0


def retrieve(query: str, filename: Optional[str] = None, top_k: int = 5) -> list:
    """Deprecated - kept for backward compatibility."""
    return []
