import sys
import io
import os
import json
import re
import base64
import time
import textwrap
from typing import Any

import pandas as pd
import gridfs
from pymongo import MongoClient

from src.logger import logging
from src.exception import CustomException


# ── MongoDB / GridFS ───────────────────────────────────────────────────────────

def get_gridfs_connection():
    """Create and return MongoDB GridFS connection."""
    try:
        mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        db = client["clarityAI_database"]
        fs = gridfs.GridFS(db)
        logging.info("MongoDB GridFS connection established.")
        return fs
    except Exception as e:
        raise CustomException(e, sys)


def fetch_csv_from_gridfs(filename: str):
    """Fetch CSV file bytes from GridFS using filename."""
    try:
        fs = get_gridfs_connection()
        grid_out = fs.find_one(
            {"filename": filename},
            sort=[("uploadDate", -1)]
        )
        if grid_out is None:
            raise Exception(f"File '{filename}' not found in GridFS")
        logging.info("CSV file '%s' fetched from GridFS.", filename)
        return grid_out.read()
    except Exception as e:
        raise CustomException(e, sys)


def load_dataframe_from_mongo(filename: str) -> pd.DataFrame:
    """Load CSV from MongoDB GridFS and return a pandas DataFrame."""
    try:
        file_bytes = fetch_csv_from_gridfs(filename)
        df = pd.read_csv(io.BytesIO(file_bytes))
        logging.info("Dataset successfully loaded into DataFrame.")
        return df
    except Exception as e:
        raise CustomException(e, sys)



# ── Chart utilities ────────────────────────────────────────────────────────────

def fig_to_b64(fig, dpi: int = 72) -> str:
    """
    Render a matplotlib Figure to a base64-encoded PNG string.
    No disk I/O — uses BytesIO only.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def analyse_chart(
    image_b64: str,
    chart_title: str,
    api_key: str,
    api_url: str,
    model: str,
) -> dict[str, Any]:
    """
    Send a base64 PNG to a vision-capable LLM endpoint via OpenAI client.
    Provider-agnostic — works with any OpenAI-compatible API (e.g. Groq)
    based on the api_key/api_url/model passed in.
    Returns a structured JSON dict with keys:
        represents, key_findings, anomalies, recommendations
    Thread-safe — no shared state.
    """
    import openai

    # Check if API key is configured
    if not api_key or api_key == "":
        logging.error("No API key provided for chart analysis. AI analysis disabled.")
        return {
            "represents": chart_title,
            "key_findings": [],
            "anomalies": [],
            "recommendations": [],
            "error": "API key not configured for chart analysis.",
        }

    system_prompt = textwrap.dedent("""
        You are a senior business/data analyst presenting a chart to decision-makers.
        Translate what the chart shows into BUSINESS insight — focus on the story,
        the drivers behind it, and what to do about it, not statistical jargon.

        For each field:
          - represents:      one plain-language sentence on what the chart shows.
          - key_findings:    the most important business takeaways and the likely
                             DRIVERS behind them (e.g. which segments/values move the
                             outcome). Quantify with the numbers visible in the chart.
          - anomalies:       outliers, imbalances, or data-quality concerns that
                             could mislead a business decision.
          - recommendations: concrete, actionable next steps a business or modelling
                             team should take based on this chart.

        Return ONLY a valid JSON object — no preamble, no markdown fences.
        Schema:
        {
          "represents":      "<one sentence: what this chart shows>",
          "key_findings":    ["<business takeaway / driver 1>", "<takeaway 2>", ...],
          "anomalies":       ["<anomaly 1>", ...],
          "recommendations": ["<actionable step 1>", ...]
        }
    """).strip()

    base_url = api_url.replace("/chat/completions", "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
            {
                "type": "text",
                "text": f"Chart title: {chart_title}\n\nAnalyse and return JSON.",
            },
        ]},
    ]

    # Retry up to 5 times on failure with exponential backoff
    for attempt in range(1, 6):
        try:
            client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=90,
            )
            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                temperature=0.3,
                messages=messages,
            )
            raw = response.choices[0].message.content.strip()
            logging.info("Vision analysis received for '%s' (attempt %d).", chart_title, attempt)
            return parse_json_response(raw)

        except Exception as exc:
            logging.warning(
                "Vision API attempt %d failed for '%s': %s",
                attempt, chart_title, exc,
            )
            if attempt == 5:
                return {
                    **empty_analysis(chart_title),
                    "error": f"API Error: {str(exc)[:100]}"
                }
            # Exponential backoff: 5s, 10s, 15s, 20s
            time.sleep(5 * attempt)

def parse_json_response(raw: str) -> dict:
    """
    Parse JSON from an LLM response with robust fallbacks.
    Handles markdown fences, control characters, and malformed JSON.
    """
    raw = raw.strip()
    
    # Remove control characters
    raw = raw.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Strip markdown code blocks
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.strip().lower().startswith("json"):
            raw = raw.strip()[4:]
    
    raw = raw.strip()
    
    # Strip <think> blocks (used by reasoning models like Qwen)
    if "<think>" in raw and "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    
    # Try direct JSON parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Try to extract valid JSON substring
        start_idx = raw.find('{')
        end_idx = raw.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                extracted = raw[start_idx:end_idx+1]
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
        
        # Fallback: regex search for JSON object (non-greedy)
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        logging.warning("JSON parse failed at pos %d: %s. Response: %s", e.pos, e.msg, raw[:300])
        return {}


def empty_analysis(title: str) -> dict[str, Any]:
    """Return a blank analysis stub used when AI analysis is skipped or fails."""
    return {
        "represents":      title,
        "key_findings":    [],
        "anomalies":       [],
        "recommendations": [],
    }