"""
chat_agent.py
==============
Handles chat. Inlines context-fetching logic from rag_pipeline.py
(which can now be deleted).
"""

import os
import sys
import io
from typing import Optional

import pandas as pd
from pymongo import MongoClient
import gridfs

from src.agents.llm_provider import call_llm
from src.logger import logging


_MONGO_URL = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_DB_NAME = "clarityAI_database"


def _get_db():
    return MongoClient(_MONGO_URL)[_DB_NAME]


def _load_df(filename: str) -> pd.DataFrame:
    db = _get_db()
    fs = gridfs.GridFS(db)
    grid_out = fs.find_one({"filename": filename}, sort=[("uploadDate", -1)])
    if grid_out is None:
        raise FileNotFoundError(f"File '{filename}' not found in GridFS.")
    return pd.read_csv(io.BytesIO(grid_out.read()))


def _get_dataset_context(filename: str) -> dict:
    try:
        df = _load_df(filename)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

        context = {
            "filename": filename,
            "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        }

        stats = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) > 0:
                stats[col] = {
                    "mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                    "min": round(float(s.min()), 4),
                    "max": round(float(s.max()), 4),
                }
        context["numeric_stats"] = stats

        from src.components.mongo_storage import get_dataset_insights
        doc = get_dataset_insights(filename)
        if doc:
            ai = doc.get("ai_insights", {})
            context["ai_summary"] = ai.get("summary", "")
            context["quality_flags"] = ai.get("quality_flags", [])

        return context
    except Exception as e:
        return {"filename": filename, "error": str(e)}


def _format_context(ctx: dict) -> str:
    if "error" in ctx:
        return f"Could not retrieve dataset: {ctx['error']}"
    lines = [
        f"Dataset: {ctx.get('filename')}",
        f"Shape: {ctx.get('shape')}",
        f"Columns: {', '.join(ctx.get('columns', [])[:10])}",
        f"Missing values: {ctx.get('missing_values')}",
        f"Duplicates: {ctx.get('duplicates')}",
    ]
    if ctx.get("numeric_stats"):
        lines.append("Key stats:")
        for col, s in list(ctx["numeric_stats"].items())[:3]:
            lines.append(f"  {col}: mean={s['mean']}, min={s['min']}, max={s['max']}")
    if ctx.get("ai_summary"):
        lines.append(f"AI Summary: {ctx['ai_summary'][:200]}")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are Vera, a helpful data science assistant.
When dataset context is provided, use it to answer accurately — never invent numbers.
Be concise and direct. If no context is provided, answer conversationally.
"""

_DATASET_KEYWORDS = {
    "column", "row", "data", "dataset", "value", "null", "missing",
    "mean", "average", "max", "min", "count", "distribution", "feature",
    "correlation", "type", "dtype", "unique", "statistic", "outlier",
    "pattern", "trend", "summary", "describe", "shape", "size",
}


def chat(
    query: str,
    filename: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> dict:
    context_str = ""
    context_available = False

    if filename and any(kw in query.lower() for kw in _DATASET_KEYWORDS):
        try:
            ctx = _get_dataset_context(filename)
            if "error" not in ctx:
                context_str = _format_context(ctx)
                context_available = True
        except Exception as e:
            logging.warning("Context fetch failed: %s", e)

    history_text = ""
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{'User' if role == 'user' else 'Vera'}: {content}\n"

    user_message = ""
    if history_text:
        user_message += f"Previous conversation:\n{history_text}\n"
    if context_str:
        user_message += f"Dataset context:\n{context_str}\n\n"
    user_message += f"Question: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message, temperature=0.2)
    except Exception as e:
        logging.error("Chat LLM call failed: %s", e)
        answer = "Sorry, something went wrong. Please try again."

    return {
        "answer": answer,
        "dataset": filename,
        "context_available": context_available,
    }