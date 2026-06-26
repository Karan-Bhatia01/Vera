"""
dtype_utils.py
==============
Bulletproofing for messy real-world dtypes.

Datasets loaded from CSV/Mongo sometimes carry pandas *extension* dtypes
(StringDtype, nullable Int64/Float64, BooleanDtype, CategoricalDtype). scikit-learn
and numpy don't understand those — e.g. `np.issubdtype(StringDtype, ...)` raises
"Cannot interpret '<StringDtype...>' as a data type", which used to crash the whole
ML run. We normalize everything back to plain numpy dtypes up front so the rest of
the pipeline only ever sees `object` / `float64` / `int64`.
"""

from __future__ import annotations

import pandas as pd


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with all pandas extension dtypes downcast to numpy.

    - nullable integer/float (Int64, Float64) → float64 (NA → NaN)
    - StringDtype / CategoricalDtype / BooleanDtype / anything else → object
    - columns that are already numpy dtypes are left untouched
    """
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if not pd.api.types.is_extension_array_dtype(dtype):
            continue
        if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype(object)
    return df
