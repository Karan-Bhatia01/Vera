import sys
import io
import os
import json
import re
import base64
import textwrap
from typing import Any

import pandas as pd
import gridfs
from pymongo import MongoClient
from groq import Groq

from src.logger import logging
from src.exception import CustomException


# ── MongoDB / GridFS ───────────────────────────────────────────────────────────

def get_gridfs_connection():
    """Create and return MongoDB GridFS connection."""
    try:
        client = MongoClient("mongodb://localhost:27017/")
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


# ── LLM (Groq) ────────────────────────────────────────────────────────────────

def llm_agent(prompt: str, role: str, context: str) -> dict:
    """
    Send a structured prompt to Groq and return the parsed JSON response.

    Returns
    -------
    dict with keys:
        "response"  → the LLM's answer string
        "metadata"  → role, original_query, context_summary
    """
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        structured_prompt = (
            f"Act as {role} and respond only in JSON format.\n"
            f"The context is: {context}\n"
            f"The user query is: {prompt}\n\n"
            "The JSON structure must always be:\n"
            "{\n"
            '    "response": "<your answer here>",\n'
            '    "metadata": {\n'
            '        "role": "<role>",\n'
            '        "original_query": "<original user query>",\n'
            '        "context_summary": "<short summary of context>"\n'
            "    }\n"
            "}"
        )

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": structured_prompt}],
            model="groq/compound",
        )

        raw_content = chat_completion.choices[0].message.content

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            return {
                "response": raw_content,
                "metadata": {
                    "role": role,
                    "original_query": prompt,
                    "context_summary": context[:100],
                }
            }

    except Exception as e:
        raise Exception(f"LLM Agent failed: {e}")


# ── Chart utilities ────────────────────────────────────────────────────────────

def fig_to_b64(fig, dpi: int = 72) -> str:
    """
    Render a matplotlib Figure to a base64-encoded PNG string.
    No disk I/O — uses BytesIO only.
    """
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def analyse_chart(
    image_b64: str,
    chart_title: str,
    oxlo_api_key: str,
    oxlo_api_url: str,
    oxlo_model: str,
) -> dict[str, Any]:
    """
    Send a base64 PNG to the Oxlo/Mistral vision endpoint via OpenAI client.
    Returns a structured JSON dict with keys:
        represents, key_findings, anomalies, recommendations
    Thread-safe — no shared state.
    """
    import openai

    system_prompt = textwrap.dedent("""
        You are a senior data analyst reviewing a chart image.
        Return ONLY a valid JSON object — no preamble, no markdown fences.
        Schema:
        {
          "represents":      "<one sentence: what this chart shows>",
          "key_findings":    ["<finding 1>", "<finding 2>", ...],
          "anomalies":       ["<anomaly 1>", ...],
          "recommendations": ["<action 1>", ...]
        }
    """).strip()

    base_url = oxlo_api_url.replace("/chat/completions", "")
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

    # Retry up to 2 times on failure
    for attempt in range(1, 3):
        try:
            client = openai.OpenAI(
                base_url=base_url,
                api_key=oxlo_api_key,
                timeout=90,
            )
            response = client.chat.completions.create(
                model=oxlo_model,
                max_tokens=512,
                messages=messages,
            )
            raw = response.choices[0].message.content.strip()
            logging.info("Oxlo analysis received for '%s' (attempt %d).", chart_title, attempt)
            return parse_json_response(raw)

        except Exception as exc:
            logging.warning(
                "Oxlo API attempt %d failed for '%s': %s",
                attempt, chart_title, exc,
            )
            if attempt == 2:
                return empty_analysis(chart_title)


def parse_json_response(raw: str) -> dict:
    """
    Parse JSON from an LLM response, stripping markdown fences if present.
    Falls back to regex extraction, then empty dict.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logging.warning("Could not parse JSON from response: %s", raw[:200])
        return {}


def empty_analysis(title: str) -> dict[str, Any]:
    """Return a blank analysis stub used when AI analysis is skipped or fails."""
    return {
        "represents":      title,
        "key_findings":    [],
        "anomalies":       [],
        "recommendations": [],
    }