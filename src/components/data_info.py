import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo


class DataInfo:
    def __init__(self, filename: str):
        self.filename = filename
        self._df = None

    def get_dataframe(self) -> pd.DataFrame:
        """Load the dataset once from GridFS and reuse it for every analysis call."""
        if self._df is None:
            self._df = load_dataframe_from_mongo(self.filename)
        return self._df

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
                "advanced_numeric": self._advanced_numeric_stats(df, numeric_cols),
                "advanced_categorical": self._advanced_categorical_stats(df, categorical_cols),
                "cardinality_flags": self._cardinality_flags(df),
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

    @staticmethod
    def _advanced_numeric_stats(df: pd.DataFrame, numeric_cols: list) -> dict:
        stats = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty: continue
            
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            outliers = int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum())
            
            stats[col] = {
                "outliers": outliers,
                "skewness": round(float(s.skew()), 3) if len(s) > 2 else 0.0,
                "kurtosis": round(float(s.kurt()), 3) if len(s) > 3 else 0.0,
                "zeros": int((s == 0).sum()),
                "negatives": int((s < 0).sum()),
            }
        return stats

    @staticmethod
    def _advanced_categorical_stats(df: pd.DataFrame, categorical_cols: list) -> dict:
        stats = {}
        for col in categorical_cols:
            s = df[col].dropna()
            if s.empty: continue
            
            try:
                is_date = pd.to_datetime(s.head(50), errors='coerce', format='mixed').notna().sum() > 25
            except Exception:
                is_date = False
                
            str_s = s.astype(str)
            lengths = str_s.str.len()
            
            stats[col] = {
                "inferred_date": bool(is_date),
                "text_length": {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "mean": round(float(lengths.mean()), 2),
                }
            }
        return stats

    @staticmethod
    def _cardinality_flags(df: pd.DataFrame) -> dict:
        flags = {"constant_columns": [], "id_columns": []}
        total = len(df)
        if total == 0: return flags
        
        nunique = df.nunique()
        flags["constant_columns"] = nunique[nunique == 1].index.tolist()
        flags["id_columns"] = nunique[nunique == total].index.tolist()
        return flags

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
