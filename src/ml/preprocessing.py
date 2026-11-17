"""
preprocessing.py
================
Turn a cleaned DataFrame + a feature plan into model-ready train/test arrays.

Steps: drop planned columns → drop rows the target can't use → encode the target →
assign each feature to numeric / ordinal / one-hot (checked against real dtypes) →
ColumnTransformer → train/test split.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.logger import logging

TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_ROWS = 20
MAX_CLASSES = 100
MAX_CATEGORIES = 20  # one-hot keeps the most common values and groups the rest as "infrequent"


def build_splits(
    df: pd.DataFrame,
    target_column: str,
    plan: dict[str, Any],
    problem_type: str,
):
    """Return (X_train, X_test, y_train, y_test, feature_names, label_encoder)."""
    drop_cols = [c for c in plan.get("drop", []) if c in df.columns and c != target_column]
    df = df.drop(columns=drop_cols)
    logging.info("Dropped columns: %s", drop_cols)

    df = _usable_rows(df, target_column, problem_type)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Class labels become 0..n-1, which every model (XGBoost included) accepts.
    label_encoder = None
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))

    numeric_cols, ordinal_cols, onehot_cols, ordinal_categories = _assign_roles(X, plan)
    if not (numeric_cols or ordinal_cols or onehot_cols):
        raise ValueError(
            "No usable feature columns are left. Every column was either dropped as an ID, "
            "free text or mostly empty, or only the target remained."
        )

    # Categories are compared as text; a missing value is just another category.
    categorical = ordinal_cols + onehot_cols
    if categorical:
        X[categorical] = X[categorical].astype(object).fillna("missing").astype(str)

    preprocessor = ColumnTransformer(
        _transformers(numeric_cols, ordinal_cols, onehot_cols, ordinal_categories),
        remainder="drop",
    )
    X_train, X_test, y_train, y_test = _split(X, y, problem_type)
    logging.info(
        "Preprocessing roles: numeric=%d, ordinal=%d, onehot=%d",
        len(numeric_cols), len(ordinal_cols), len(onehot_cols),
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    logging.info("Preprocessed: X_train=%s, X_test=%s, features=%d",
                 X_train.shape, X_test.shape, len(feature_names))
    return X_train, X_test, y_train, y_test, feature_names, label_encoder


def _usable_rows(df: pd.DataFrame, target: str, problem_type: str) -> pd.DataFrame:
    """Drop rows without a target, and classes too small to learn from. Raise if nothing sensible is left."""
    missing = int(df[target].isna().sum())
    if missing:
        df = df[df[target].notna()]
        logging.warning("Dropped %d rows with a missing target value.", missing)

    if problem_type == "classification":
        if df[target].nunique() > MAX_CLASSES:
            raise ValueError(
                f"Target column '{target}' has more than {MAX_CLASSES} different values. "
                "It looks like an identifier or free text; pick a column with fewer categories."
            )
        counts = df[target].value_counts()
        rare = counts[counts < 2].index
        if len(rare):
            df = df[~df[target].isin(rare)]
            logging.warning("Dropped classes with a single row: %s", list(rare))

    if len(df) < MIN_ROWS:
        raise ValueError(f"Only {len(df)} usable rows remain; at least {MIN_ROWS} are needed to train.")
    if df[target].nunique() < 2:
        raise ValueError(f"Target column '{target}' has only one value, so there is nothing to predict.")
    return df


def _assign_roles(X: pd.DataFrame, plan: dict[str, Any]):
    """Bucket each feature into numeric / ordinal / one-hot.

    Trusts the plan, but checks it against real dtypes: a text column the LLM called
    numeric is one-hot encoded instead, so it can never reach StandardScaler.
    """
    ordinal_plan = plan.get("ordinal") or {}
    planned_onehot = set(plan.get("onehot") or [])
    numeric, ordinal, onehot = [], [], []

    for col in X.columns:
        is_number = pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col])
        if not is_number and isinstance(ordinal_plan.get(col), list):
            ordinal.append(col)
        elif is_number and col not in planned_onehot:
            numeric.append(col)
        else:
            onehot.append(col)

    # Explicit order from the plan, as text, without duplicates (OrdinalEncoder rejects repeats).
    ordinal_categories = [list(dict.fromkeys(str(v) for v in ordinal_plan[col])) for col in ordinal]
    return numeric, ordinal, onehot, ordinal_categories


def _transformers(numeric_cols, ordinal_cols, onehot_cols, ordinal_categories):
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                # add_indicator: "was this value missing?" is often a useful signal on its own.
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if ordinal_cols:
        transformers.append((
            "ord",
            OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value",
                           unknown_value=-1, dtype=float),
            ordinal_cols,
        ))
    if onehot_cols:
        transformers.append((
            "ohe",
            OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=MAX_CATEGORIES,
                          sparse_output=False),
            onehot_cols,
        ))
    return transformers


def _split(X, y, problem_type: str):
    """80/20 split, stratified for classification whenever every class can be represented."""
    if problem_type == "classification":
        try:
            return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        except ValueError:
            logging.warning("Too few rows per class to stratify; using a plain random split.")
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
