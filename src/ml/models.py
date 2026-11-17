"""
models.py
=========
The model registry: maps a problem type to the candidate estimators we can train.
Defaults are tuned to work well on typical tabular data without a search.
XGBoost / LightGBM are included only when their packages are installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB = True
except ImportError:
    _XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _LGBM = True
except ImportError:
    _LGBM = False

RANDOM_STATE = 42
BOOSTERS = ["LightGBM", "XGBoost", "GradientBoosting"]  # strongest first


def get_models(problem_type: str, log_target: bool = False) -> dict[str, Any]:
    """Return {model_name: fresh_estimator} for the given problem type.

    For regression every model learns a standardised target (log-transformed first when
    `log_target` is set), so scale-sensitive models like SVR, KNN and Lasso work on any
    target range. Predictions are converted back to the original units automatically.
    """
    if problem_type == "classification":
        return _classifiers()
    return {name: _scaled_target(model, log_target) for name, model in _regressors().items()}


def _classifiers() -> dict[str, Any]:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "DecisionTree":       DecisionTreeClassifier(min_samples_leaf=5, random_state=RANDOM_STATE),
        "RandomForest":       RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        # "GradientBoosting" uses scikit-learn's histogram booster: faster and usually stronger.
        "GradientBoosting":   HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, early_stopping=True, random_state=RANDOM_STATE),
        "SVM":                SVC(probability=True, random_state=RANDOM_STATE),
        "KNN":                KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "NaiveBayes":         GaussianNB(),
    }
    if _XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
        )
    if _LGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)
    return models


def _regressors() -> dict[str, Any]:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso":            Lasso(alpha=0.001, max_iter=5000, random_state=RANDOM_STATE),
        "DecisionTree":     DecisionTreeRegressor(min_samples_leaf=5, random_state=RANDOM_STATE),
        "RandomForest":     RandomForestRegressor(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, early_stopping=True, random_state=RANDOM_STATE),
        "SVR":              SVR(C=3.0),
        "KNN":              KNeighborsRegressor(n_neighbors=15, weights="distance"),
    }
    if _XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0,
        )
    if _LGBM:
        models["LightGBM"] = LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)
    return models


def _scaled_target(model, log_target: bool) -> TransformedTargetRegressor:
    steps = [FunctionTransformer(np.log1p, inverse_func=np.expm1, check_inverse=False)] if log_target else []
    return TransformedTargetRegressor(regressor=model, transformer=make_pipeline(*steps, StandardScaler()))
