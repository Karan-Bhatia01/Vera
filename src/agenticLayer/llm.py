from __future__ import annotations

import json
import os
import sys
from typing import Any

from groq import Groq
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo

load_dotenv()

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_LLM_MODEL = "groq/compound"
_LLM_MAX_TOKENS = 2_048
_UNIQUE_PREVIEW_LIMIT = 10

_SYSTEM_PROMPT = """\
You are a senior data analyst reviewing a dataset summary.
You must respond ONLY with a valid JSON object — no preamble, no markdown fences.

Strict rules:
1. Do NOT recalculate or re-derive any numbers — treat all statistics as ground truth.
2. Interpret and explain what the statistics *mean* for real-world use.
3. Flag data quality concerns: high null rates, suspicious duplicates, skewed distributions.
4. Note uncertainty where statistics alone are insufficient to draw conclusions.
5. Keep tone professional but accessible.

Return exactly this JSON shape (all fields required):
{
  "summary": "<2-3 sentence plain-English overview of the dataset>",
  "quality_flags": [
    {
      "column": "<column name or 'dataset'>",
      "severity": "<high | medium | low>",
      "issue": "<short label>",
      "detail": "<1-2 sentence explanation>"
    }
  ],
  "column_insights": [
    {
      "column": "<column name>",
      "insight": "<1-2 sentence interpretation of this column's distribution or values>"
    }
  ],
  "next_steps": [
    {
      "title": "<short action title>",
      "detail": "<1-2 sentence explanation of why and how>"
    }
  ],
  "uncertainty_notes": "<paragraph noting what cannot be concluded from statistics alone>"
}
"""


# ──────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────
class AnalysisExplainer:
    """
    Loads a dataset from MongoDB GridFS, computes descriptive statistics
    using pandas, and generates structured JSON insights via Groq LLM.

    Usage
    -----
    explainer = AnalysisExplainer("my_dataset.csv")
    result = explainer.run()
    # result["analysis"]    → dict of computed statistics
    # result["unique"]      → unique value previews dict
    # result["ai_insights"] → structured dict (summary, quality_flags, etc.)
    """

    def __init__(self, filename: str) -> None:
        try:
            self.filename = filename
            self.df = self._load_dataframe()
            self.client = Groq(api_key=self._require_env("GROQ_API_KEY"))
            logging.info("AnalysisExplainer initialised for file: %s", filename)
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── private helpers ────────────────────────

    @staticmethod
    def _require_env(key: str) -> str:
        """Return env variable *key* or raise a clear error if missing."""
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{key}' is not set."
            )
        return value

    def _load_dataframe(self):
        """Load dataset from MongoDB GridFS via utility helper."""
        try:
            df = load_dataframe_from_mongo(self.filename)
            logging.info("DataFrame loaded: %d rows × %d columns.", *df.shape)
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── public methods ─────────────────────────

    def compute_analysis(self) -> dict[str, Any]:
        """
        Compute descriptive statistics using pandas only.
        The LLM never performs calculations — all numbers originate here.

        Returns
        -------
        dict with keys:
            shape, columns, dtypes, null_values, null_percentages,
            duplicate_rows, numeric_columns, categorical_columns,
            describe, memory_usage_mb, unique_counts, sample_rows
        """
        try:
            df = self.df
            total_rows = len(df)

            null_counts: dict[str, int] = df.isnull().sum().to_dict()
            null_pct: dict[str, float] = {
                col: round(count / total_rows * 100, 2)
                for col, count in null_counts.items()
            }

            analysis: dict[str, Any] = {
                "shape": df.shape,
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / 1024 ** 2, 3
                ),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
                "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
                "null_values": null_counts,
                "null_percentages": null_pct,
                "duplicate_rows": int(df.duplicated().sum()),
                "describe": df.describe(include="all").to_dict(),
                "unique_counts": df.nunique().to_dict(),
                "sample_rows": df.head(5).to_dict(orient="records"),
            }

            logging.info("Dataset analysis computed successfully.")
            return analysis

        except Exception as e:
            raise CustomException(e, sys) from e

    def unique_preview(
        self, limit: int = _UNIQUE_PREVIEW_LIMIT
    ) -> dict[str, dict[str, Any]]:
        """
        Return a preview of unique values for every column.

        Returns
        -------
        dict: column → {"values": [...], "total_unique": int, "truncated": bool}
        """
        try:
            preview: dict[str, dict[str, Any]] = {}
            for col in self.df.columns:
                unique_vals = self.df[col].dropna().unique()
                total = len(unique_vals)
                preview[col] = {
                    "values": unique_vals[:limit].tolist(),
                    "total_unique": total,
                    "truncated": total > limit,
                }
            logging.info("Unique value preview computed.")
            return preview
        except Exception as e:
            raise CustomException(e, sys) from e

    def explain_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """
        Send pre-computed statistics to the LLM.

        Returns a structured dict with keys:
            summary, quality_flags, column_insights, next_steps, uncertainty_notes

        The LLM is strictly instructed to return valid JSON only.
        Falls back to a plain error dict if JSON parsing fails.
        """
        try:
            # Build condensed prompt — exclude raw describe/sample to save tokens
            _EXCLUDE = {"describe", "sample_rows"}
            prompt_data = {k: v for k, v in analysis.items() if k not in _EXCLUDE}

            if "describe" in analysis:
                prompt_data["describe_summary"] = {
                    col: {
                        stat: round(val, 4) if isinstance(val, float) else val
                        for stat, val in stats.items()
                        if stat in ("count", "mean", "std", "min", "50%", "max")
                    }
                    for col, stats in analysis["describe"].items()
                }

            user_message = (
                f"Dataset statistics:\n\n"
                f"{json.dumps(prompt_data, indent=2, default=str)}\n\n"
                "Analyse these and return the JSON insights object."
            )

            completion = self.client.chat.completions.create(
                model=_LLM_MODEL,
                max_tokens=_LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )

            raw: str = completion.choices[0].message.content.strip()

            # Strip accidental markdown fences if the model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            insights: dict[str, Any] = json.loads(raw)
            logging.info("Structured AI insights generated successfully.")
            return insights

        except json.JSONDecodeError as exc:
            logging.warning("LLM returned invalid JSON: %s", exc)
            return {
                "summary": "AI response could not be parsed as structured JSON.",
                "quality_flags": [],
                "column_insights": [],
                "next_steps": [],
                "uncertainty_notes": raw,
            }
        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self) -> dict[str, Any]:
        """
        Execute the full pipeline: load → analyse → explain.

        Returns
        -------
        dict with keys:
            "analysis"     → raw statistics dict
            "unique"       → unique value previews dict
            "ai_insights"  → structured JSON dict from LLM
        """
        try:
            logging.info("Starting analysis pipeline for '%s'.", self.filename)
            analysis = self.compute_analysis()
            unique = self.unique_preview()
            ai_insights = self.explain_analysis(analysis)

            logging.info("Pipeline completed successfully.")
            return {
                "analysis": analysis,
                "unique": unique,
                "ai_insights": ai_insights,
            }

        except Exception as e:
            raise CustomException(e, sys) from e