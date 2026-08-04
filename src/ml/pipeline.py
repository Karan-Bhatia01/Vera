"""
pipeline.py
===========
MLPipeline — the orchestrator. Loads the data, asks the agents for a feature
plan and a model shortlist, then delegates preprocessing, training, explanation,
and persistence to the focused modules in this package.

Public surface is unchanged from the old src.components.ml_pipeline:
    MLPipeline(filename, target_column).run(progress_callback) -> dict
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any, Callable

from pymongo import MongoClient

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo
from src.agents.feature_engineering_agent import decide_feature_plan
from src.agents.model_selection_agent import select_models

from src.ml.dtype_utils import normalize_dtypes
from src.ml.problem_type import detect_problem_type
from src.ml.models import get_models
from src.ml.preprocessing import build_splits
from src.ml.training import train_and_evaluate, best_model_name
from src.ml.persistence import save_results

_MONGO_URL = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_DB_NAME = "clarityAI_database"


class MLPipeline:
    """Fully automatic, LLM-guided ML pipeline."""

    def __init__(self, filename: str, target_column: str) -> None:
        try:
            self.filename = filename
            self.target_column = target_column
            self.db = MongoClient(_MONGO_URL)[_DB_NAME]
            # normalize_dtypes up front so no downstream step ever meets an
            # extension dtype (StringDtype/Int64) that numpy/sklearn can't read.
            self.df = normalize_dtypes(load_dataframe_from_mongo(filename))
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
            X_train, X_test, y_train, y_test, feature_names, _le = build_splits(
                self.df, self.target_column, feature_plan, problem_type
            )

            progress(28, "Selecting candidate models...")
            models = self._shortlist_models(problem_type, X_train, y_train)

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
            best_model = best_model_name(results, rank_metric)

            progress(90, "Finalizing results...")
            mongo_id = save_results(
                self.db, self.filename, self.target_column, results,
                problem_type, feature_plan, rank_metric, best_model,
            )

            progress(100, "Pipeline complete!")
            logging.info("=== ML Pipeline complete. Best: %s ===", best_model)
            return self._response(
                results, problem_type, feature_plan, best_model, rank_metric, mongo_id
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── helpers ────────────────────────────────────────────────────────────────
    def _shortlist_models(self, problem_type: str, X_train, y_train) -> dict[str, Any]:
        """Let the agent pick 3-5 models from the full registry."""
        all_models = get_models(problem_type)
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
        models = {name: all_models[name] for name in selected if name in all_models}
        logging.info("Agent selected %d models: %s", len(models), list(models))
        return models or all_models  # fall back to all if selection came back empty

    def _response(self, results, problem_type, feature_plan, best_model, rank_metric, mongo_id) -> dict:
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
            "mongo_doc_id":  mongo_id,
        }
