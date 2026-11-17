"""
Decide whether the target column is a classification or regression problem.
"""

from __future__ import annotations

import pandas as pd

from src.logger import logging

# int columns with <= this many distinct values are treated as class labels
CLASS_UNIQUE_MAX = 12


def detect_problem_type(series: pd.Series) -> str:
    """Return 'classification' or 'regression' for the given target column.

    Rules (in order):
      - bool / text / categorical                       classification
      - whole numbers (int, or floats like 0.0 / 1.0)
        with <= CLASS_UNIQUE_MAX distinct values        classification
      - any other number                                regression
      - anything else                                   classification (safe default)
    """
    dtype = series.dtype
    n_unique = int(series.nunique())

    if (
        pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    ):
        problem = "classification"
    elif pd.api.types.is_numeric_dtype(dtype):
        values = series.dropna()
        whole_numbers = (values == values.round()).all()
        problem = "classification" if whole_numbers and n_unique <= CLASS_UNIQUE_MAX else "regression"
    else:
        problem = "classification"

    logging.info("Problem type: %s (dtype=%s, nunique=%d)", problem, dtype, n_unique)
    return problem
