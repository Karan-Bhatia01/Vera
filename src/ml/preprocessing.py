"""
preprocessing.py
================
Turn a raw DataFrame + a feature plan into model-ready train/test arrays.

Steps: drop planned columns → split features/target → encode target →
assign each feature to numeric / ordinal / one-hot (validated against real
dtypes) → cap one-hot feature explosion → ColumnTransformer → train/test split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder,
)
from sklearn.compose import ColumnTransformer

from src.logger import logging

TEST_SIZE = 0.2
RANDOM_STATE = 42
_MAX_ONEHOT_FEATURES = 50   # cap to avoid memory blow-up from high-cardinality cols
_AUTO_ONEHOT_MAX_CARD = 10  # unplanned object cols above this are dropped, not encoded


def build_splits(
    df: pd.DataFrame,
    target_column: str,
    plan: dict[str, Any],
    problem_type: str,
):
    """Return (X_train, X_test, y_train, y_test, feature_names, label_encoder)."""
    df = df.copy()

    drop_cols = [c for c in plan.get("drop", []) if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logging.info("Dropped columns: %s", drop_cols)

    X = df.drop(columns=[target_column])
    y = df[target_column].copy()

    # Encode a categorical target to integer labels.
    label_encoder = None
    if problem_type == "classification" and y.dtype == object:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    numeric_cols, ordinal_cols, onehot_cols, ordinal_categories = _assign_roles(X, plan)

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if ordinal_cols:
        transformers.append((
            "ord",
            OrdinalEncoder(categories=ordinal_categories,
                           handle_unknown="use_encoded_value",
                           unknown_value=-1, dtype=float),
            ordinal_cols,
        ))
    if onehot_cols:
        transformers.append((
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            onehot_cols,
        ))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if problem_type == "classification" else None,
    )
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


def _assign_roles(X: pd.DataFrame, plan: dict[str, Any]):
    """Bucket each feature column into numeric / ordinal / one-hot.

    Trusts the plan but validates against real dtypes, so a text column the
    LLM mislabelled as numeric can never reach StandardScaler.
    Mutates `X` in place when dropping unusable columns.
    """
    ordinal_cols = [c for c in plan.get("ordinal", {}) if c in X.columns]
    onehot_cols = [c for c in plan.get("onehot", []) if c in X.columns]
    numeric_cols = [c for c in plan.get("numeric", []) if c in X.columns]

    # Infer roles for any column the plan didn't mention.
    planned = set(ordinal_cols + onehot_cols + numeric_cols)
    for col in list(X.columns):
        if col in planned:
            continue
        if X[col].dtype == object:
            if X[col].nunique() <= _AUTO_ONEHOT_MAX_CARD:
                onehot_cols.append(col)
            else:
                X.drop(columns=[col], inplace=True)  # high-cardinality text → drop
        else:
            numeric_cols.append(col)

    # Guard: a column the plan called numeric that isn't actually numeric would
    # crash StandardScaler (float('Petrol')). Reroute it instead.
    validated_numeric = []
    for col in numeric_cols:
        if col not in X.columns:
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            validated_numeric.append(col)
        elif X[col].nunique() <= _AUTO_ONEHOT_MAX_CARD:
            onehot_cols.append(col)
            logging.warning("Column '%s' planned numeric but is text — routing to one-hot.", col)
        else:
            X.drop(columns=[col], inplace=True)
            logging.warning("Column '%s' planned numeric but is high-cardinality text — dropping.", col)
    numeric_cols = validated_numeric

    onehot_cols = _cap_onehot([c for c in dict.fromkeys(onehot_cols) if c in X.columns], X)
    ordinal_cols = [c for c in ordinal_cols if c in X.columns]
    ordinal_categories = _ordinal_categories(ordinal_cols, plan)

    return numeric_cols, ordinal_cols, onehot_cols, ordinal_categories


def _cap_onehot(onehot_cols: list[str], X: pd.DataFrame) -> list[str]:
    """Keep lowest-cardinality one-hot columns so total features stay bounded."""
    total = sum(X[c].nunique() for c in onehot_cols)
    if total <= _MAX_ONEHOT_FEATURES:
        return onehot_cols

    logging.warning("One-hot would create %d features — capping to %d.",
                    total, _MAX_ONEHOT_FEATURES)
    kept, cumulative = [], 0
    for col, card in sorted(((c, X[c].nunique()) for c in onehot_cols), key=lambda t: t[1]):
        if cumulative + card <= _MAX_ONEHOT_FEATURES:
            kept.append(col)
            cumulative += card
        else:
            X.drop(columns=[col], inplace=True)
            logging.warning("Dropping one-hot column '%s' (%d uniques) to save memory.", col, card)
    return kept


def _ordinal_categories(ordinal_cols: list[str], plan: dict[str, Any]):
    """Build sklearn's `categories` arg: explicit orders where given, else 'auto'."""
    categories, has_explicit = [], False
    for col in ordinal_cols:
        cats = plan.get("ordinal", {}).get(col, "auto")
        if isinstance(cats, list):
            categories.append(cats)
            has_explicit = True
        else:
            categories.append(None)
    return categories if has_explicit else "auto"
