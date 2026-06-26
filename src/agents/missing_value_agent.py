"""
missing_value_agent.py
========================
Replaces get_ai_insights() in eda_processing.py.

Computes null stats + dtype + skew in Python, sends ONE prompt to Groq.
Returns {} immediately if no missing values — no API call at all.
"""

import sys
import json

import pandas as pd

from src.agents.llm_provider import call_llm, parse_json_from_response
from src.logger import logging
from src.exception import CustomException


VALID_METHODS = ["mean", "median", "mode", "ffill", "bfill", "zero", "drop"]

SYSTEM_PROMPT = """\
You are a data analyst deciding how to fill missing values.
For each column provided, choose exactly one fill method:
mean, median, mode, ffill, bfill, zero, drop

Rules:
- Numeric, roughly symmetric → mean
- Numeric, skewed or outliers → median
- Categorical/text → mode
- Sequential/time data → ffill or bfill
- Missing = 0 makes sense (counts, flags) → zero
- Over 60% missing → drop

Return ONLY valid JSON, no explanation:
{"column_name": "method", ...}
"""


def decide_missing_value_strategy(df: pd.DataFrame) -> dict:
    """
    Returns {column: method} for every column with missing values.
    Returns {} immediately if no missing values — no API call.
    """
    try:
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].index.tolist()

        if not cols_with_nulls:
            logging.info("No missing values — skipping missing value agent.")
            return {}

        # Build compact per-column stats
        col_stats = []
        for col in cols_with_nulls:
            series = df[col]
            entry = {
                "col": col,
                "dtype": str(series.dtype),
                "null_pct": round(float(series.isnull().mean() * 100), 1),
            }
            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 1:
                    entry["skew"] = round(float(non_null.skew()), 2)
            else:
                top = series.dropna().value_counts().head(3)
                entry["top_vals"] = list(top.index.astype(str))
            col_stats.append(entry)

        user_prompt = (
            f"Columns with missing values:\n"
            f"{json.dumps(col_stats, separators=(',', ':'))}\n\n"
            "Return fill strategy JSON."
        )

        raw = call_llm(SYSTEM_PROMPT, user_prompt)
        strategy = parse_json_from_response(raw)
        if not isinstance(strategy, dict):
            strategy = {}

        # Validate methods, fallback on anything invalid
        for col in cols_with_nulls:
            if strategy.get(col) not in VALID_METHODS:
                fallback = "median" if pd.api.types.is_numeric_dtype(df[col]) else "mode"
                strategy[col] = fallback

        logging.info("Missing value strategy: %s", strategy)
        return strategy

    except Exception as e:
        raise CustomException(e, sys) from e