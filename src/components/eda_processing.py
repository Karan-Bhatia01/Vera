from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo

# Chart types that just count the values of one column.
_COUNT_CHARTS = {"bar", "pie", "doughnut", "radar", "polararea"}


def _all_finite(datasets: list[dict]) -> bool:
    """True if every number in the chart data (plain values or x/y/r points) is finite."""
    for ds in datasets:
        for value in ds["data"]:
            numbers = value.values() if isinstance(value, dict) else [value]
            if not all(np.isfinite(n) for n in numbers):
                return False
    return True


class DataPreprocessing:
    """
    Preprocessing + EDA pipeline.

    1. preprocess_data(strategy)                → cleaned DataFrame (in-memory only)
    2. generate_eda_report(df, recommended)     → Chart.js-ready chart data, one per AI-recommended chart
    """

    def __init__(
        self,
        filename: str,
        target_column: str,
        columns_to_drop: list[str] | None = None,
    ) -> None:
        try:
            self.filename        = filename
            self.target_column   = target_column
            self.columns_to_drop = columns_to_drop or []
            self.df = load_dataframe_from_mongo(filename)
            logging.info("DataPreprocessing loaded '%s' — shape %s", filename, self.df.shape)

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def preprocess_data(self, strategy: dict[str, str] | None = None) -> pd.DataFrame:
        """Cleans data: removes duplicates, drops columns, fills missing values."""
        try:
            df = self.df.copy()
            df.drop_duplicates(inplace=True)

            if self.columns_to_drop:
                df.drop(columns=[c for c in self.columns_to_drop if c in df.columns], inplace=True)

            if df.isnull().values.any() and strategy:
                for col, method in strategy.items():
                    if col not in df.columns: continue

                    if method in ("mean", "median", "zero"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    if method == "mean":     df[col] = df[col].fillna(df[col].mean() or 0)
                    elif method == "median": df[col] = df[col].fillna(df[col].median() or 0)
                    elif method == "mode":   df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)
                    elif method == "ffill":  df[col] = df[col].ffill()
                    elif method == "bfill":  df[col] = df[col].bfill()
                    elif method == "zero":   df[col] = df[col].fillna(0)
                    elif method == "drop":   df.dropna(subset=[col], inplace=True)
                    else:                    df[col] = df[col].fillna(method)

            logging.info("Preprocessing complete. Final shape: %s", df.shape)
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── EDA report ────────────────────────────────────────────────────────────
    def generate_eda_report(self, df: pd.DataFrame, recommended_charts: list[dict]) -> list[dict]:
        """Build Chart.js data for every chart the AI recommended. Bad charts are skipped."""
        charts = []
        for rec in recommended_charts:
            try:
                chart = self._build_chart(df, rec)
                if chart:
                    charts.append(chart)
            except Exception as e:
                logging.warning("Skipping chart %s: %s", rec, e)

        logging.info("EDA report built — %d charts.", len(charts))
        return charts

    @staticmethod
    def _numeric_sample(df: pd.DataFrame, cols: list[str], max_points: int) -> pd.DataFrame:
        """Numeric-only copy of `cols` with NaNs dropped, sampled down for the browser."""
        data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        return data.sample(min(max_points, len(data)))

    def _build_chart(self, df: pd.DataFrame, rec: dict) -> dict | None:
        """Aggregate the raw data for one recommended chart, or return None if it can't be drawn."""
        chart_type = rec.get("chart_type", "bar").lower()
        cols = rec.get("columns", [])

        # Every column the chart needs must exist (scatter/line need 2, bubble needs 3).
        needed = {"scatter": 2, "line": 2, "bubble": 3}.get(chart_type, 1)
        if len(cols) < needed or any(c not in df.columns for c in cols[:needed]):
            return None

        x = cols[0]
        labels, datasets = [], []

        if chart_type in _COUNT_CHARTS:
            counts = df[x].value_counts().head(10)
            labels = counts.index.astype(str).tolist()
            datasets = [{"label": "Count", "data": counts.values.tolist()}]

        elif chart_type == "histogram":
            values = pd.to_numeric(df[x], errors="coerce").dropna()
            if values.empty:
                return None
            hist, bins = np.histogram(values, bins=20)
            labels = [f"{bins[i]:.2f} - {bins[i + 1]:.2f}" for i in range(len(hist))]
            datasets = [{"label": "Frequency", "data": hist.tolist()}]

        elif chart_type == "scatter":
            y = cols[1]
            sample = self._numeric_sample(df, [x, y], max_points=500)
            points = [{"x": float(px), "y": float(py)} for px, py in zip(sample[x], sample[y])]
            datasets = [{"label": f"{y} vs {x}", "data": points}]

        elif chart_type == "bubble":
            y, size = cols[1], cols[2]
            sample = self._numeric_sample(df, [x, y, size], max_points=200)
            # Scale bubble radius into a readable 5–20px range.
            low, high = sample[size].min(), sample[size].max()
            spread = (high - low) or 1
            points = [
                {"x": float(px), "y": float(py), "r": 5 + 15 * (float(ps) - low) / spread}
                for px, py, ps in zip(sample[x], sample[y], sample[size])
            ]
            datasets = [{"label": f"{y} vs {x} (size: {size})", "data": points}]

        elif chart_type == "line":
            y = cols[1]
            # Groups whose values are all missing average to NaN — drop them.
            averages = df.groupby(x)[y].mean().dropna().head(20)
            labels = averages.index.astype(str).tolist()
            datasets = [{"label": f"Average {y}", "data": averages.values.tolist()}]

        if not datasets or not datasets[0]["data"] or not _all_finite(datasets):
            # Nothing to draw, or NaN/inf that would make the response invalid JSON.
            return None

        return {
            "chart_type": chart_type,
            "columns": cols,
            "labels": labels,
            "datasets": datasets,
            "takeaway": rec.get("story_takeaway", ""),
        }
