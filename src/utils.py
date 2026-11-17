import io
import json
import re
import sys

import pandas as pd

from src.core.connections import get_gridfs
from src.logger import logging
from src.exception import CustomException


# ── MongoDB / GridFS ───────────────────────────────────────────────────────────

def fetch_csv_from_gridfs(filename: str) -> bytes:
    """Fetch the newest stored copy of a CSV file from GridFS."""
    try:
        grid_out = get_gridfs().find_one({"filename": filename}, sort=[("uploadDate", -1)])
        if grid_out is None:
            raise Exception(f"File '{filename}' not found in GridFS")
        logging.info("CSV file '%s' fetched from GridFS.", filename)
        return grid_out.read()
    except Exception as e:
        raise CustomException(e, sys)


def read_csv_robust(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse CSV bytes, trying common encodings and delimiters.
    Keeps the parse with the MOST columns (best delimiter); skips bad lines.
    Used for both upload validation and analysis, so they always agree.
    """
    file_bytes = file_bytes.replace(b"\x00", b"")  # NULL bytes confuse the C parser
    best_df = None

    for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        for sep in [",", "\t", ";", "|"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding,
                    sep=sep,
                    on_bad_lines="skip",
                    engine="c",
                    skip_blank_lines=True,
                )
            except Exception:
                continue
            if df.shape[0] >= 1 and df.shape[1] >= 1 and (best_df is None or df.shape[1] > best_df.shape[1]):
                best_df = df

        # A multi-column parse with this encoding is good enough.
        if best_df is not None and best_df.shape[1] > 1:
            break

    if best_df is None:
        raise ValueError("Could not parse the uploaded file as a valid CSV.")
    logging.info("CSV parsed — shape=%s", best_df.shape)
    return best_df


def load_dataframe_from_mongo(filename: str) -> pd.DataFrame:
    """Load CSV from MongoDB GridFS and return a pandas DataFrame."""
    try:
        df = read_csv_robust(fetch_csv_from_gridfs(filename))
        logging.info("Dataset successfully loaded into DataFrame.")
        return df
    except Exception as e:
        raise CustomException(e, sys)


# ── LLM response parsing ───────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict | list:
    """
    Parse JSON from an LLM response with robust fallbacks.
    Handles markdown fences, <think> blocks, and JSON wrapped in extra text.
    Returns {} if nothing parseable is found.
    """
    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.strip().lower().startswith("json"):
            raw = raw.strip()[4:]
    raw = raw.strip()

    # Strip <think> blocks (used by reasoning models like Qwen)
    if "<think>" in raw and "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Try the outermost {...} or [...] substring
        for start_c, end_c in [("{", "}"), ("[", "]")]:
            start_idx, end_idx = raw.find(start_c), raw.rfind(end_c)
            if start_idx != -1 and end_idx > start_idx:
                try:
                    return json.loads(raw[start_idx:end_idx + 1])
                except json.JSONDecodeError:
                    pass

        # Last resort: the first small {...} object
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logging.warning("JSON parse failed at pos %d: %s. Response: %s", e.pos, e.msg, raw[:300])
        return {}
