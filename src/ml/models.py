"""
models.py
=========
The model registry: maps a problem type to the candidate estimators we can train.
XGBoost / LightGBM are included only when their packages are installed.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

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


def get_models(problem_type: str) -> dict[str, Any]:
    """Return {model_name: fresh_estimator} for the given problem type."""
    if problem_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "DecisionTree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
            "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "GradientBoosting":   GradientBoostingClassifier(random_state=RANDOM_STATE),
            "SVM":                SVC(probability=True, random_state=RANDOM_STATE),
            "KNN":                KNeighborsClassifier(),
            "NaiveBayes":         GaussianNB(),
        }
        if _XGB:
            models["XGBoost"] = XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
            )
        if _LGBM:
            models["LightGBM"] = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    else:  # regression
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge":            Ridge(random_state=RANDOM_STATE),
            "Lasso":            Lasso(random_state=RANDOM_STATE),
            "DecisionTree":     DecisionTreeRegressor(random_state=RANDOM_STATE),
            "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "SVR":              SVR(),
            "KNN":              KNeighborsRegressor(),
        }
        if _XGB:
            models["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
        if _LGBM:
            models["LightGBM"] = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)

    return models
