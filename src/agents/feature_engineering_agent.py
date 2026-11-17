"""
feature_engineering_agent.py
=============================
Computes all column stats in Python first, then sends ONE compact prompt
to Groq with the full picture — no tool-call loops, no multi-turn reasoning.
"""

import sys
import json
from pathlib import Path

import pandas as pd

from src.agents.llm_provider import call_llm
from src.utils import parse_json_response
from src.logger import logging
from src.exception import CustomException

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "feature_engineering.md").read_text(encoding="utf-8")


def _heuristic_plan(df: pd.DataFrame, target_column: str) -> dict:
    """Dtype-based feature plan used when the LLM is unavailable.

    IDs and free text were already removed by src.ml.cleaning, so numbers stay numeric
    and every text column is one-hot encoded (rare values are grouped by the encoder).
    """
    plan = {"drop": [], "ordinal": {}, "onehot": [], "numeric": []}
    for col in df.columns:
        if col == target_column:
            continue
        role = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "onehot"
        plan[role].append(col)
    return plan


def decide_feature_plan(df: pd.DataFrame, target_column: str) -> dict:
    """
    Returns {"drop": [], "ordinal": {}, "onehot": [], "numeric": []},
    the shape src.ml.preprocessing.build_splits expects.
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
            plan = parse_json_response(raw)
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

        # Safety net: any unclassified column → numeric or one-hot
        all_cols = set(c for c in df.columns if c != target_column)
        classified = (
            set(plan["drop"]) |
            set(plan["onehot"]) |
            set(plan["numeric"]) |
            set(plan["ordinal"].keys())
        )
        for col in all_cols - classified:
            plan["numeric" if pd.api.types.is_numeric_dtype(df[col]) else "onehot"].append(col)

        logging.info("Feature plan: %s", plan)
        return plan

    except Exception as e:
        raise CustomException(e, sys) from e