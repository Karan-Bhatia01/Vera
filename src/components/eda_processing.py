from __future__ import annotations

import os
import sys
import json
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException
from src.utils import (
    load_dataframe_from_mongo,
    fig_to_b64,
    analyse_chart,
    empty_analysis,
)

load_dotenv()

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

_VISION_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_VISION_API_KEY = os.getenv("GROQ_API_KEY", "")
# Multimodal model for chart analysis. We use Qwen 3.6 27B since the Llama 3.2 Vision models 
# were decommissioned. Overridable via GROQ_VISION_MODEL.
_VISION_MODEL   = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
_CHART_DPI      = 72


class DataPreprocessing:
    """
    Full preprocessing + EDA pipeline.

    Workflow
    --------
    1. get_ai_insights()          → {col: method} — skips agent if no nulls
    2. preprocess_data(strategy)  → cleaned DataFrame (in-memory only)
    3. generate_eda_report(df)    → {chart_title: image_b64}
                                    No AI calls here — done on-demand via /analyse_chart
    4. analyse_single(b64, title) → AI analysis dict for one chart
    """

    def __init__(
        self,
        filename: str,
        target_column: str,
        columns_to_drop: list[str] | None = None,
        vision_api_key: str = "",
    ) -> None:
        try:
            self.filename        = filename
            self.target_column   = target_column
            self.columns_to_drop = columns_to_drop or []
            self._vision_key     = vision_api_key or _VISION_API_KEY

            mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
            client  = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            self.db = client["clarityAI_database"]
            self.fs = gridfs.GridFS(self.db)

            self.df = load_dataframe_from_mongo(filename)
            logging.info("DataPreprocessing loaded '%s' — shape %s", filename, self.df.shape)

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _valid_hue(self, df: pd.DataFrame) -> str | None:
        """Returns the target column if it exists in the dataframe for color mapping."""
        tc = self.target_column
        return tc if isinstance(tc, str) and tc.strip() and tc in df.columns else None

    def _save_chart(self, fig, title: str, charts: dict) -> None:
        """Helper to convert a plot to base64 and save it, then clean up memory."""
        charts[title] = fig_to_b64(fig, _CHART_DPI)
        plt.close("all")

    def _get_top_features(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Finds the most important numeric and categorical columns to plot."""
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        top_num = []
        if num_cols:
            if self.target_column in num_cols:
                corrs = df[num_cols].corr()[self.target_column].abs().drop(self.target_column, errors="ignore")
                top_num = [self.target_column] + corrs.sort_values(ascending=False).head(4).index.tolist()
            else:
                top_num = df[num_cols].var().sort_values(ascending=False).head(5).index.tolist()

        top_cat = []
        if cat_cols:
            cat_nunique = df[cat_cols].nunique()
            top_cat = cat_nunique[cat_nunique > 1].sort_values().head(3).index.tolist()

        return top_num, top_cat

    # ── 1. Missing-value strategy (agent-based) ───────────────────────────────


    # ── Preprocessing ──────────────────────────────────────────────────────
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

    # ── 3. EDA report — charts only, no AI calls ───────────────────────────────
    def generate_eda_report(self, df: pd.DataFrame) -> dict[str, str]:
        """
        Build all charts and return {title: image_b64}.
        AI analysis is NOT done here — it happens on-demand via /analyse_chart.
        Page loads instantly.
        """
        try:
            charts = self._build_all_charts(df)
            if charts is None:
                logging.warning("_build_all_charts returned None — using empty dict.")
                charts = {}
            logging.info("EDA report built — %d charts (no AI calls).", len(charts))
            return charts
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── 4. Analyse a single chart on demand ────────────────────────────────────
    def analyse_single(self, image_b64: str, chart_title: str) -> dict[str, Any]:
        """Called by /analyse_chart route for one chart at a time."""
        try:
            result = analyse_chart(
                image_b64, chart_title,
                self._vision_key, _VISION_API_URL, _VISION_MODEL,
            )
            logging.info("On-demand analysis done for '%s'.", chart_title)
            return result
        except Exception as e:
            logging.warning("On-demand analysis failed for '%s': %s", chart_title, e)
            return empty_analysis(chart_title)

    # ── Chart builder ──────────────────────────────────────────────────────────
    def _build_all_charts(self, df: pd.DataFrame) -> dict[str, str]:
        """Generates up to 10 meaningful charts using the top features."""
        charts: dict[str, str] = {}
        try:
            top_num, top_cat = self._get_top_features(df)

            # 1. Distributions (up to 4)
            for col in top_num[:4]:
                if len(charts) >= 10: break
                title = f"Distribution — {col}"
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
                ax.set_title(title); ax.set_xlabel(col)
                self._save_chart(fig, title, charts)

            # 2. Boxplots (up to 2)
            for col in top_num[:2]:
                if len(charts) >= 10: break
                title = f"Boxplot — {col}"
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(y=df[col].dropna(), ax=ax, color="#55A868")
                ax.set_title(title)
                self._save_chart(fig, title, charts)

            # 3. Value Counts (up to 3)
            for col in top_cat:
                if len(charts) >= 10: break
                title = f"Value Counts — {col}"
                vc = df[col].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=vc.index, y=vc.values, ax=ax, hue=vc.index, palette="muted", legend=False)
                ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel("Count")
                plt.xticks(rotation=35, ha="right"); plt.tight_layout()
                self._save_chart(fig, title, charts)

            # 4. Correlation Heatmap (1)
            if len(top_num) >= 2 and len(charts) < 10:
                title = "Correlation Heatmap"
                corr = df[top_num].corr()
                fig, ax = plt.subplots(figsize=(max(6, len(top_num)), max(5, len(top_num) - 1)))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax, square=True)
                ax.set_title(title); plt.tight_layout()
                self._save_chart(fig, title, charts)

            # 5. Scatter Plots (up to 2)
            if len(top_num) >= 2:
                corr_matrix = df[top_num].corr().abs()
                pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False).head(2)
                for (col_x, col_y), _ in pairs.items():
                    if len(charts) >= 10: break
                    title = f"Scatter — {col_x} vs {col_y}"
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.scatterplot(data=df, x=col_x, y=col_y, hue=self._valid_hue(df), alpha=0.65, ax=ax)
                    ax.set_title(title); plt.tight_layout()
                    self._save_chart(fig, title, charts)
                    
        except Exception as e:
            logging.warning("Error building charts: %s", e)

        return charts