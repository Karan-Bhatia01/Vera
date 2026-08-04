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
# Fast multimodal model for chart analysis. qwen3.6 is a *reasoning* model that
# emits <think> blocks and is slow for this — Llama 4 Scout is vision-capable and
# much quicker. Overridable via GROQ_VISION_MODEL.
_VISION_MODEL   = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")
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

    # ── helpers ───────────────────────────────────────────────────────────────
    def _valid_hue(self, df: pd.DataFrame) -> str | None:
        tc = self.target_column
        if isinstance(tc, str) and tc.strip() and tc in df.columns.tolist():
            return tc
        return None

    # ── 1. Missing-value strategy (agent-based) ───────────────────────────────


    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    def preprocess_data(
        self,
        missing_value_strategy: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        try:
            df = self.df.copy()
            logging.info("Preprocessing started.")

            before = len(df)
            df.drop_duplicates(inplace=True)
            logging.info("Duplicates removed: %d rows dropped.", before - len(df))

            if self.columns_to_drop:
                df.drop(
                    columns=[c for c in self.columns_to_drop if c in df.columns],
                    inplace=True,
                )
                logging.info("Dropped columns: %s", self.columns_to_drop)

            if df.isnull().values.any():
                strategy = missing_value_strategy or {}
                for col, method in strategy.items():
                    if col not in df.columns:
                        continue
                    if method in ("mean", "median", "zero"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    if method == "mean":
                        fill_val = df[col].mean()
                        df[col] = df[col].fillna(fill_val if pd.notna(fill_val) else 0)
                    elif method == "median":
                        fill_val = df[col].median()
                        df[col] = df[col].fillna(fill_val if pd.notna(fill_val) else 0)
                    elif method == "mode":
                        mode_vals = df[col].mode()
                        if not mode_vals.empty:
                            df[col] = df[col].fillna(mode_vals.iloc[0])
                    elif method == "ffill":
                        df[col] = df[col].ffill()
                    elif method == "bfill":
                        df[col] = df[col].bfill()
                    elif method == "zero":
                        df[col] = df[col].fillna(0)
                    elif method == "drop":
                        df.dropna(subset=[col], inplace=True)
                    else:
                        df[col] = df[col].fillna(method)
                logging.info("Missing values handled.")
            else:
                logging.info("No missing values — skipping fill step.")

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
        charts: dict[str, str] = {}
        try:
            numeric_cols     = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
            
            # ── Select Top Features ──
            top_numeric = []
            if numeric_cols:
                if self.target_column in numeric_cols:
                    # Sort by absolute correlation with target
                    corrs = df[numeric_cols].corr()[self.target_column].abs().drop(self.target_column, errors="ignore")
                    top_numeric = [self.target_column] + corrs.sort_values(ascending=False).head(4).index.tolist()
                else:
                    # Sort by variance
                    var = df[numeric_cols].var().sort_values(ascending=False)
                    top_numeric = var.head(5).index.tolist()

            top_categorical = []
            if categorical_cols:
                # Rank categorical features by lowest cardinality > 1
                cat_nunique = df[categorical_cols].nunique()
                cat_nunique = cat_nunique[cat_nunique > 1]
                top_categorical = cat_nunique.sort_values().head(3).index.tolist()

        except Exception as e:
            logging.warning("Could not determine column types: %s", e)
            return charts

        # 1. Distributions (up to 4)
        for col in top_numeric[:4]:
            if len(charts) >= 10: break
            try:
                title = f"Distribution — {col}"
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
                ax.set_title(title); ax.set_xlabel(col)
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped '%s': %s", title, e)
            finally:
                plt.close("all")

        # 2. Boxplots (up to 2)
        for col in top_numeric[:2]:
            if len(charts) >= 10: break
            try:
                title = f"Boxplot — {col}"
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(y=df[col].dropna(), ax=ax, color="#55A868")
                ax.set_title(title)
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped '%s': %s", title, e)
            finally:
                plt.close("all")

        # 3. Value Counts (up to 3)
        for col in top_categorical:
            if len(charts) >= 10: break
            try:
                title = f"Value Counts — {col}"
                vc = df[col].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=vc.index, y=vc.values, ax=ax,
                            hue=vc.index, palette="muted", legend=False)
                ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel("Count")
                plt.xticks(rotation=35, ha="right"); plt.tight_layout()
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped '%s': %s", title, e)
            finally:
                plt.close("all")

        # 4. Correlation Heatmap (1)
        if len(top_numeric) >= 2 and len(charts) < 10:
            try:
                title = "Correlation Heatmap"
                corr  = df[top_numeric].corr()
                n     = len(top_numeric)
                fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                            linewidths=0.5, ax=ax, square=True)
                ax.set_title(title); plt.tight_layout()
                charts[title] = fig_to_b64(fig, _CHART_DPI)
            except Exception as e:
                logging.warning("Skipped 'Correlation Heatmap': %s", e)
            finally:
                plt.close("all")

        # 5. Scatter Plots (up to 2)
        if len(top_numeric) >= 2:
            corr_matrix = df[top_numeric].corr().abs()
            pairs = (
                corr_matrix
                .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                .stack().sort_values(ascending=False).head(2)
            )
            for (col_x, col_y), _ in pairs.items():
                if len(charts) >= 10: break
                try:
                    title = f"Scatter — {col_x} vs {col_y}"
                    hue   = self._valid_hue(df)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.scatterplot(data=df, x=col_x, y=col_y,
                                    hue=hue, alpha=0.65, ax=ax)
                    ax.set_title(title); plt.tight_layout()
                    charts[title] = fig_to_b64(fig, _CHART_DPI)
                except Exception as e:
                    logging.warning("Skipped '%s': %s", title, e)
                finally:
                    plt.close("all")

        return charts