import sys
import io
import os
import pandas as pd

import gridfs
from pymongo import MongoClient

from src.logger import logging
from src.exception import CustomException


class DataInfo:
    def __init__(self, filename: str):
        try:
            self.filename = filename

            mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            self.db = client["clarityAI_database"]
            self.fs = gridfs.GridFS(self.db)

            logging.info("MongoDB connection established in Data-Info.")

        except Exception as e:
            raise CustomException(e, sys)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Fetch file from Database using filename and return as pandas DataFrame
        """
        try:
            # Get latest file with given filename
            grid_out = self.fs.find_one(
                {"filename": self.filename},
                sort=[("uploadDate", -1)]
            )

            if grid_out is None:
                raise Exception("File not found in MongoDB GridFS")

            file_bytes = grid_out.read()
            df = pd.read_csv(io.BytesIO(file_bytes))

            logging.info("Dataset successfully loaded into DataFrame.")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def dataset_analysis(self) -> dict:
        """
        Perform basic dataset analysis and return results
        """
        try:
            df = self.get_dataframe()

            total_rows = len(df) or 1  # avoid divide-by-zero on empty frames
            null_counts = df.isnull().sum().to_dict()
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

            analysis = {
                "shape": df.shape,
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / 1024 ** 2, 3
                ),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_values": null_counts,
                "null_percentages": {
                    col: round(count / total_rows * 100, 2)
                    for col, count in null_counts.items()
                },
                "unique_counts": df.nunique().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "describe": df.describe().to_dict(),
                # Richer, display-ready stats (all cast to native Python types
                # so jsonify never chokes on numpy scalars).
                "numeric_stats": self._numeric_stats(df, numeric_cols),
                "top_categories": self._top_categories(df, categorical_cols),
                "correlations": self._correlations(df, numeric_cols),
            }

            logging.info("Dataset analysis completed.")
            return analysis

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _numeric_stats(df: pd.DataFrame, numeric_cols: list) -> dict:
        """Per-column min/mean/median/max/std for numeric columns."""
        stats = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            stats[col] = {
                "min":    round(float(s.min()), 3),
                "max":    round(float(s.max()), 3),
                "mean":   round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "std":    round(float(s.std()), 3) if len(s) > 1 else 0.0,
            }
        return stats

    @staticmethod
    def _top_categories(df: pd.DataFrame, categorical_cols: list, top_n: int = 5) -> dict:
        """Top-N most frequent values (with counts) for each categorical column."""
        top = {}
        for col in categorical_cols:
            vc = df[col].value_counts().head(top_n)
            top[col] = [{"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()]
        return top

    @staticmethod
    def _correlations(df: pd.DataFrame, numeric_cols: list, limit: int = 10) -> list:
        """Strongest pairwise correlations among numeric columns, sorted by |r|."""
        if len(numeric_cols) < 2:
            return []
        corr = df[numeric_cols].corr(numeric_only=True)
        pairs = []
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                val = corr.loc[a, b]
                if pd.notna(val):
                    pairs.append({"columns": [a, b], "correlation": round(float(val), 3)})
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        return pairs[:limit]

    def get_unique_column_values(self) -> dict:
        """
        Returns a dictionary with column names and limited unique values.
        """
        try:
            df = self.get_dataframe()
            unique_val = {}

            for col in df.columns:
                values = df[col].dropna().unique().tolist()

                unique_val[col] = {
                    "values": values[:10],
                    "total_unique": len(values),
                    "truncated": len(values) > 10
                }

            return unique_val
        except Exception as e:
            raise CustomException(e, sys)
