"""
training.py
===========
Fit every candidate model, evaluate it, and collect feature importances.
Also builds an ensemble that averages the strongest models.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error, confusion_matrix,
)

from src.logger import logging

_TOP_FEATURES = 15
_ENSEMBLE_SIZE = 3
ENSEMBLE_NAME = "Ensemble"


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
    failures: dict[str, str] = {}
    total = len(models) or 1

    for i, (name, model) in enumerate(models.items()):
        if progress:
            progress(30 + int((i / total) * 50), f"Training {name}... ({i + 1}/{total})")
        try:
            model.fit(X_train, y_train)
            results[name] = {
                "metrics": evaluate(model, X_test, y_test, problem_type),
                "feature_importance": _feature_importance(model, X_test, y_test, feature_names),
                "model_object": model,
            }
            logging.info("Trained %s: %s", name, results[name]["metrics"])
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            logging.warning("Model %s failed: %s", name, exc, exc_info=True)

    if not results and failures:
        # Surface the real reason instead of a generic "nothing trained" message.
        # Failures are usually identical across models, so the distinct set is short.
        reasons = "; ".join(sorted({f"{n} → {r}" for n, r in failures.items()}))
        raise RuntimeError(f"All {len(failures)} candidate models failed to train. Reasons: {reasons}")

    return results


def add_ensemble(results: dict[str, Any], X_test, y_test, problem_type: str, rank_metric: str) -> None:
    """Add an "Ensemble" entry that averages the top models. Averaging usually beats any single one."""
    ranked = sorted(results, key=lambda n: results[n]["metrics"].get(rank_metric, float("-inf")), reverse=True)
    members = [results[n]["model_object"] for n in ranked[:_ENSEMBLE_SIZE]]
    if len(members) < 2:
        return
    if problem_type == "classification" and not all(hasattr(m, "predict_proba") for m in members):
        return

    ensemble = AveragedModel(members, problem_type)
    results[ENSEMBLE_NAME] = {
        "metrics": evaluate(ensemble, X_test, y_test, problem_type),
        # Explain the ensemble through its strongest member.
        "feature_importance": results[ranked[0]]["feature_importance"],
        "model_object": ensemble,
    }
    logging.info("Ensemble of %s: %s", ranked[:_ENSEMBLE_SIZE], results[ENSEMBLE_NAME]["metrics"])


class AveragedModel:
    """Averages already-trained models: class probabilities for classification, values for regression."""

    def __init__(self, models: list, problem_type: str):
        self.models = models
        self.problem_type = problem_type

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def predict(self, X):
        if self.problem_type == "classification":
            return self.predict_proba(X).argmax(axis=1)  # labels are 0..n-1
        return np.mean([m.predict(X) for m in self.models], axis=0)


def best_model_name(results: dict[str, Any], rank_metric: str) -> str:
    """Name of the model with the highest primary metric."""
    return max(results, key=lambda n: results[n]["metrics"].get(rank_metric, float("-inf")))


def evaluate(model, X_test, y_test, problem_type: str) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    if problem_type == "classification":
        return _classification_metrics(model, X_test, y_test, y_pred)
    return _regression_metrics(y_test, y_pred)


def _classification_metrics(model, X_test, y_test, y_pred) -> dict[str, Any]:
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }
    try:  # ROC-AUC: binary directly, multi-class as one-vs-rest
        y_prob = model.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        else:
            metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"), 4)
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


def _feature_importance(model, X_test, y_test, feature_names: list[str]) -> dict[str, float]:
    """Top features by tree importance, linear coefficient size, or (for any other model) permutation."""
    inner = getattr(model, "regressor_", model)  # look inside the target-scaling wrapper
    if hasattr(inner, "feature_importances_"):
        values = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        coef = np.atleast_2d(inner.coef_)
        # One row keeps its sign (direction of the effect); multi-class averages the sizes.
        values = coef[0] if coef.shape[0] == 1 else np.abs(coef).mean(axis=0)
    else:
        # How much the score drops when a feature is shuffled; works for every model.
        rows = min(len(y_test), 1000)
        values = permutation_importance(
            model, X_test[:rows], np.asarray(y_test)[:rows], n_repeats=3, random_state=0,
        ).importances_mean

    paired = {name: round(float(v), 6) for name, v in zip(feature_names, values)}
    return dict(sorted(paired.items(), key=lambda kv: abs(kv[1]), reverse=True)[:_TOP_FEATURES])
