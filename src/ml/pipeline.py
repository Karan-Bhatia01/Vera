"""
pipeline.py
===========
MLPipeline — the orchestrator. Loads and cleans the data, asks the agents for a
feature plan and a model shortlist, then delegates preprocessing and training to
the focused modules in this package.

    MLPipeline(filename, target_column).run(progress_callback) -> dict
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any, Callable

import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo
from src.agents.feature_engineering_agent import decide_feature_plan
from src.agents.model_selection_agent import select_models

from src.ml.cleaning import clean_dataset
from src.ml.dtype_utils import normalize_dtypes
from src.ml.problem_type import detect_problem_type
from src.ml.models import BOOSTERS, get_models
from src.ml.preprocessing import build_splits
from src.ml.training import add_ensemble, best_model_name, train_and_evaluate

# Kernel models get very slow on large data, so they're skipped above this many training rows.
_KERNEL_MODEL_MAX_ROWS = 10_000
_KERNEL_MODELS = {"SVM", "SVR"}


class MLPipeline:
    """Fully automatic, LLM-guided ML pipeline."""

    def __init__(self, filename: str, target_column: str) -> None:
        try:
            self.filename = filename
            self.target_column = str(target_column).strip()
            # normalize_dtypes up front so no downstream step ever meets an
            # extension dtype (StringDtype/Int64) that numpy/sklearn can't read;
            # clean_dataset turns number-like text and dates into usable features.
            raw = normalize_dtypes(load_dataframe_from_mongo(filename))
            raw.columns = [str(c).strip() for c in raw.columns]
            if self.target_column not in raw.columns:
                raise ValueError(f"Target column '{self.target_column}' was not found in the dataset.")
            self.df = clean_dataset(raw, self.target_column)
            logging.info("MLPipeline loaded '%s' — shape %s", filename, self.df.shape)
        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self, progress_callback: Callable[[int, str], None] | None = None) -> dict[str, Any]:
        def progress(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logging.info("[%d%%] %s", pct, msg)

        try:
            logging.info("=== ML Pipeline starting for '%s' ===", self.filename)

            progress(5, "Asking the agent for a feature engineering plan...")
            feature_plan = decide_feature_plan(self.df, self.target_column)

            progress(15, "Detecting problem type...")
            problem_type = detect_problem_type(self.df[self.target_column])

            progress(25, "Preprocessing data...")
            X_train, X_test, y_train, y_test, feature_names, _ = build_splits(
                self.df, self.target_column, feature_plan, problem_type
            )

            progress(28, "Selecting candidate models...")
            log_target = problem_type == "regression" and _is_skewed(self.df[self.target_column])
            models = self._shortlist_models(problem_type, X_train, y_train, log_target)

            results = train_and_evaluate(
                models, X_train, X_test, y_train, y_test,
                feature_names, problem_type, progress=progress,
            )
            if not results:
                # train_and_evaluate raises with concrete reasons when models fail;
                # reaching here means no candidate models were produced at all.
                raise Exception(
                    "No candidate models were available to train for "
                    f"problem type '{problem_type}'."
                )

            rank_metric = "accuracy" if problem_type == "classification" else "r2"
            progress(85, "Combining the strongest models...")
            add_ensemble(results, X_test, y_test, problem_type, rank_metric)
            best_model = best_model_name(results, rank_metric)

            progress(90, "Finalizing results...")
            progress(100, "Pipeline complete!")
            logging.info("=== ML Pipeline complete. Best: %s ===", best_model)
            return self._response(results, problem_type, feature_plan, best_model, rank_metric)

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── helpers ────────────────────────────────────────────────────────────────
    def _shortlist_models(self, problem_type: str, X_train, y_train, log_target: bool) -> dict[str, Any]:
        """Let the agent pick 3-5 models, then make sure the shortlist is strong and fast enough."""
        all_models = get_models(problem_type, log_target)
        target_balance = None
        if problem_type == "classification":
            labels = y_train.tolist() if hasattr(y_train, "tolist") else list(y_train)
            target_balance = dict(Counter(labels))

        selected = select_models(
            problem_type=problem_type,
            n_rows=X_train.shape[0],
            n_features=X_train.shape[1],
            target_balance=target_balance,
        )
        # Always include a random forest and the best available booster: on tabular data
        # they're the most reliable performers, whatever the agent picked.
        booster = next(name for name in BOOSTERS if name in all_models)
        names = list(dict.fromkeys([*selected, "RandomForest", booster]))
        if X_train.shape[0] > _KERNEL_MODEL_MAX_ROWS:
            names = [n for n in names if n not in _KERNEL_MODELS]

        models = {name: all_models[name] for name in names if name in all_models}
        logging.info("Training %d models: %s", len(models), list(models))
        return models

    def _response(self, results, problem_type, feature_plan, best_model, rank_metric) -> dict:
        # Strip the (non-serialisable) estimator objects; sort by primary metric.
        clean = {
            name: {"metrics": data["metrics"], "feature_importance": data.get("feature_importance")}
            for name, data in results.items()
        }
        clean = dict(sorted(
            clean.items(),
            key=lambda kv: kv[1]["metrics"].get(rank_metric, float("-inf")),
            reverse=True,
        ))
        return {
            "filename":      self.filename,
            "target_column": self.target_column,
            "problem_type":  problem_type,
            "feature_plan":  feature_plan,
            "results":       clean,
            "best_model":    best_model,
            "rank_metric":   rank_metric,
        }


def _is_skewed(target: pd.Series) -> bool:
    """A non-negative target with a long right tail (prices, counts, incomes) is learned on a log scale."""
    values = pd.to_numeric(target, errors="coerce").dropna()
    return bool(len(values) > 2 and values.min() >= 0 and values.skew() > 1)
