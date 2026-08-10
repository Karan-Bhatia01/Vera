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
      - bool / object / categorical        classification
      - float                              regression
      - int, <= CLASS_UNIQUE_MAX uniques   classification
      - int, more uniques                  regression
      - anything else                      classification (safe default)
    """
    dtype = series.dtype
    n_unique = int(series.nunique())

    if (
        pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    ):
        problem = "classification"
    elif pd.api.types.is_float_dtype(dtype):
        problem = "regression"
    elif pd.api.types.is_integer_dtype(dtype):
        problem = "classification" if n_unique <= CLASS_UNIQUE_MAX else "regression"
    else:
        problem = "classification"

    logging.info("Problem type: %s (dtype=%s, nunique=%d)", problem, dtype, n_unique)
    return problem
