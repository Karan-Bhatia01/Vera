"""
feature_engineering_agent.py
=============================
Replaces ml_pipeline.py's llm_feature_plan().

Computes all column stats in Python first, then sends ONE compact prompt
to Groq with the full picture — no tool-call loops, no multi-turn reasoning.
Typical response time: 1-3 seconds instead of 30-60+.
"""

import sys
import json

import pandas as pd

from src.agents.llm_provider import call_llm, parse_json_from_response
from src.logger import logging
from src.exception import CustomException


SYSTEM_PROMPT = """\
You are a machine learning feature engineer. You will receive a JSON summary
of a dataset's columns and return a feature engineering plan.

Rules:
- Drop ID-like columns (unique value count == row count), names, or free text.
- Numeric continuous columns → "numeric".
- Categorical with natural order (e.g. low/medium/high) → "ordinal". Include the order list.
- Categorical with no natural order and cardinality <= 15 → "onehot".
- Categorical with cardinality > 15 → "drop".
- Never include the target column in any list.

Return ONLY valid JSON, no explanation, no markdown:
{"drop": ["col1"], "ordinal": {"col2": ["low","medium","high"]}, "onehot": ["col3"], "numeric": ["col4"]}
"""


def _heuristic_plan(df: pd.DataFrame, target_column: str) -> dict:
    """Dtype/cardinality-based feature plan used when the LLM is unavailable.

    Mirrors the rules in SYSTEM_PROMPT: numeric → numeric (ID-like dropped),
    low-cardinality text → one-hot, high-cardinality text → drop.
    """
    plan = {"drop": [], "ordinal": {}, "onehot": [], "numeric": []}
    n_rows = len(df)
    for col in df.columns:
        if col == target_column:
            continue
        series = df[col]
        n_unique = int(series.nunique())
        if pd.api.types.is_numeric_dtype(series):
            # Unique-per-row numeric is almost certainly an ID → drop.
            if n_rows and n_unique == n_rows:
                plan["drop"].append(col)
            else:
                plan["numeric"].append(col)
        elif n_unique <= 15:
            plan["onehot"].append(col)
        else:
            plan["drop"].append(col)
    return plan


def decide_feature_plan(df: pd.DataFrame, target_column: str) -> dict:
    """
    Returns {"drop": [], "ordinal": {}, "onehot": [], "numeric": []}
    in the exact shape ml_pipeline.py already expects.
    """
    try:
        # Build compact column summary — all stats computed in Python, not by LLM
        col_summary = []
        for col in df.columns:
            if col == target_column:
                continue
            series = df[col]
            entry = {
                "name": col,
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique()),
                "null_pct": round(float(series.isnull().mean() * 100), 1),
                "sample": [str(v) for v in series.dropna().unique()[:6].tolist()],
            }
            col_summary.append(entry)

        user_prompt = (
            f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns. "
            f"Target column: '{target_column}'.\n\n"
            f"Column stats:\n{json.dumps(col_summary, separators=(',', ':'))}\n\n"
            "Return the feature engineering plan as JSON."
        )

        # The LLM is best-effort. If it's rate-limited / times out / returns
        # garbage, fall back to a dtype-based plan so training still runs
        # instead of failing the whole job.
        try:
            raw = call_llm(SYSTEM_PROMPT, user_prompt)
            plan = parse_json_from_response(raw)
            if not isinstance(plan, dict) or not plan:
                raise ValueError("empty/invalid plan from LLM")
        except Exception as llm_err:
            logging.warning(
                "Feature-plan LLM unavailable (%s) — using heuristic fallback.", llm_err
            )
            plan = _heuristic_plan(df, target_column)

        # Ensure expected keys exist
        plan.setdefault("drop", [])
        plan.setdefault("ordinal", {})
        plan.setdefault("onehot", [])
        plan.setdefault("numeric", [])

        # Safety net: any unclassified column → numeric or drop
        all_cols = set(c for c in df.columns if c != target_column)
        classified = (
            set(plan["drop"]) |
            set(plan["onehot"]) |
            set(plan["numeric"]) |
            set(plan["ordinal"].keys())
        )
        for col in all_cols - classified:
            if df[col].dtype == object:
                plan["drop"].append(col)
            else:
                plan["numeric"].append(col)

        logging.info("Feature plan: %s", plan)
        return plan

    except Exception as e:
        raise CustomException(e, sys) from e