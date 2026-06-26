"""
training.py
===========
Fit every candidate model, evaluate it, and collect feature importances.
A single implementation used by the orchestrator (previously this logic was
duplicated inside ml_pipeline.py).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error, confusion_matrix,
)

from src.logger import logging

_TOP_FEATURES = 15


def train_and_evaluate(
    models: dict[str, Any],
    X_train, X_test, y_train, y_test,
    feature_names: list[str],
    problem_type: str,
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Train each model and return {name: {metrics, feature_importance, model_object}}.

    A model that fails to fit is skipped (logged), never aborting the run.
    """
    results: dict[str, Any] = {}
    total = len(models) or 1

    for i, (name, model) in enumerate(models.items()):
        if progress:
            progress(30 + int((i / total) * 50), f"Training {name}... ({i + 1}/{total})")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = (
                _classification_metrics(model, X_test, y_test, y_pred)
                if problem_type == "classification"
                else _regression_metrics(y_test, y_pred)
            )
            results[name] = {
                "metrics": metrics,
                "feature_importance": _feature_importance(model, feature_names),
                "model_object": model,
            }
            logging.info("Trained %s: %s", name, metrics)
        except Exception as exc:
            logging.warning("Model %s failed: %s", name, exc)

    return results


def best_model_name(results: dict[str, Any], rank_metric: str) -> str:
    """Name of the model with the highest primary metric."""
    return max(results, key=lambda n: results[n]["metrics"].get(rank_metric, float("-inf")))


def _classification_metrics(model, X_test, y_test, y_pred) -> dict[str, Any]:
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }
    try:  # ROC-AUC only defined for binary
        y_prob = model.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
    except Exception:
        pass
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metrics


def _regression_metrics(y_test, y_pred) -> dict[str, Any]:
    return {
        "r2":   round(r2_score(y_test, y_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae":  round(mean_absolute_error(y_test, y_pred), 4),
    }


def _feature_importance(model, feature_names: list[str]) -> dict[str, float] | None:
    """Top features by tree importance or linear coefficient magnitude."""
    if hasattr(model, "feature_importances_"):
        values = [round(float(v), 6) for v in model.feature_importances_]
        key = lambda kv: kv[1]
    elif hasattr(model, "coef_"):
        coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        values = [round(float(v), 6) for v in coef]
        key = lambda kv: abs(kv[1])
    else:
        return None

    paired = dict(zip(feature_names, values))
    return dict(sorted(paired.items(), key=key, reverse=True)[:_TOP_FEATURES])
