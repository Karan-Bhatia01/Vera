"""
cleaning.py
===========
Make a raw CSV usable for modelling before any feature planning happens:

- text that is really numbers ("$1,200", "45%", " 3.5 ") becomes numeric
- date columns become year / month / day / weekday numbers
- columns that can't help a model are dropped: constant, mostly empty,
  row identifiers, and free text with a different value on almost every row
"""

from __future__ import annotations

import re

import pandas as pd

from src.logger import logging

_PARSE_SHARE = 0.9      # a text column converts when this share of its values parse
_MAX_MISSING = 0.8      # columns emptier than this are dropped
_FREE_TEXT_SHARE = 0.5  # text with more distinct values than this share of rows is ID-like
_NUMBER_JUNK = re.compile(r"[\s,$€£₹%]")
_ID_SUFFIX = re.compile(r"(Id|ID|_id)$")  # CustomerId, UserID, user_id (but not "Paid")
_ID_NAMES = {"id", "index", "row", "rownumber", "row_number"}


def clean_dataset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return a cleaned copy of `df`. The target is only converted to numbers, never dropped."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if _is_text(df[col]):
            df[col] = _maybe_numeric(df[col])

    features = [c for c in df.columns if c != target]
    for col in features:
        if _is_text(df[col]) and _looks_like_dates(df[col]):
            df = _expand_date(df, col)

    useless = [c for c in df.columns if c != target and _is_useless(c, df[c])]
    if useless:
        logging.info("Dropping columns that can't help a model: %s", useless)
    return df.drop(columns=useless)


def _is_text(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _maybe_numeric(series: pd.Series) -> pd.Series:
    """Convert text like "$1,200" or "45%" to numbers when nearly every value parses."""
    values = series.dropna().astype(str)
    if values.empty:
        return series
    parsed = pd.to_numeric(values.str.replace(_NUMBER_JUNK, "", regex=True), errors="coerce")
    if parsed.notna().mean() < _PARSE_SHARE:
        return series
    cleaned = series.astype(str).str.replace(_NUMBER_JUNK, "", regex=True)
    return pd.to_numeric(cleaned.where(series.notna()), errors="coerce")


def _looks_like_dates(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= _PARSE_SHARE


def _expand_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replace a date column with the parts models can use."""
    dates = pd.to_datetime(df[col], errors="coerce", format="mixed")
    df[f"{col}_year"] = dates.dt.year
    df[f"{col}_month"] = dates.dt.month
    df[f"{col}_day"] = dates.dt.day
    df[f"{col}_weekday"] = dates.dt.dayofweek
    logging.info("Expanded date column '%s' into year/month/day/weekday.", col)
    return df.drop(columns=[col])


def _is_useless(name: str, series: pd.Series) -> bool:
    n_rows = len(series)
    n_unique = series.nunique(dropna=True)
    if n_unique <= 1 or series.isna().mean() > _MAX_MISSING:
        return True
    if pd.api.types.is_integer_dtype(series) and n_unique == n_rows:
        return _is_row_id(name, series)
    # Text that is different on most rows (names, IDs, comments) can't be encoded usefully.
    return _is_text(series) and n_unique > max(50, _FREE_TEXT_SHARE * n_rows)


def _is_row_id(name: str, series: pd.Series) -> bool:
    """A unique whole-number column is an ID if it counts rows (0,1,2,…) or is named like one.

    Unique amounts such as prices are kept: they are neither consecutive nor named "...Id".
    """
    consecutive = series.max() - series.min() + 1 == len(series)
    named_like_id = bool(_ID_SUFFIX.search(name)) or name.lower() in _ID_NAMES
    return consecutive or named_like_id
