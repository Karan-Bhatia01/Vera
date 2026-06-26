"""
model_selection_agent.py
==========================
Selects 3-5 models to train instead of always training all 9+.

Computes dataset stats in Python, sends ONE prompt to Groq.
Returns a list of model names that are valid keys in ml_pipeline._get_models().
"""

import sys
import json
from collections import Counter

import numpy as np

from src.agents.llm_provider import call_llm, parse_json_from_response
from src.logger import logging
from src.exception import CustomException


CLASSIFICATION_MODELS = [
    "LogisticRegression", "DecisionTree", "RandomForest",
    "GradientBoosting", "SVM", "KNN", "NaiveBayes", "XGBoost", "LightGBM",
]

REGRESSION_MODELS = [
    "LinearRegression", "Ridge", "Lasso", "DecisionTree",
    "RandomForest", "GradientBoosting", "SVR", "KNN", "XGBoost", "LightGBM",
]

SYSTEM_PROMPT = """\
You are an ML engineer selecting which models to train given dataset characteristics.
Choose 3-5 models from the available list. Return ONLY a JSON array, no explanation:
["Model1", "Model2", "Model3"]

Guidance:
- Small datasets (<2000 rows): prefer simpler models (LogisticRegression, Ridge, DecisionTree, NaiveBayes)
- Large datasets (>10000 rows): tree ensembles work well (RandomForest, GradientBoosting, XGBoost, LightGBM)
- High feature count (>50): avoid SVM and KNN (slow, poor scaling)
- Imbalanced classes: prefer GradientBoosting, XGBoost, RandomForest
- Always include at least one fast baseline (LogisticRegression for classification, Ridge for regression)
- Never return more than 5 or fewer than 3 models
"""


def select_models(
    problem_type: str,
    n_rows: int,
    n_features: int,
    target_balance: dict | None = None,
) -> list[str]:
    """
    Returns list of 3-5 model name strings, valid keys in ml_pipeline._get_models().
    """
    try:
        available = (
            CLASSIFICATION_MODELS if problem_type == "classification"
            else REGRESSION_MODELS
        )

        stats = {
            "problem_type": problem_type,
            "n_rows": n_rows,
            "n_features": n_features,
            "available_models": available,
        }

        if problem_type == "classification" and target_balance:
            counts = list(target_balance.values()) if isinstance(target_balance, dict) else []
            if counts:
                min_c, max_c = min(counts), max(counts)
                stats["class_imbalance_ratio"] = round(max_c / max(min_c, 1), 2)

        user_prompt = (
            f"Dataset stats:\n{json.dumps(stats, separators=(',', ':'))}\n\n"
            "Choose 3-5 models. Return JSON array only."
        )

        # Best-effort: if the LLM is rate-limited / unavailable, fall through
        # with an empty selection — the 3-5 backfill below picks sensible
        # defaults so training never fails just because model selection did.
        try:
            raw = call_llm(SYSTEM_PROMPT, user_prompt)
            selected = parse_json_from_response(raw)
        except Exception as llm_err:
            logging.warning(
                "Model-selection LLM unavailable (%s) — using default models.", llm_err
            )
            selected = []

        if not isinstance(selected, list):
            selected = []

        # Filter to valid names only
        selected = [m for m in selected if m in available]

        # Enforce 3-5 range
        if len(selected) < 3:
            for fallback in available:
                if fallback not in selected:
                    selected.append(fallback)
                if len(selected) >= 3:
                    break

        selected = selected[:5]
        logging.info("Model selection: %s", selected)
        return selected

    except Exception as e:
        raise CustomException(e, sys) from e