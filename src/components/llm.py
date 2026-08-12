from __future__ import annotations

import sys
import time
import json
from typing import Any

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo

from src.components.crew_config import build_analysis_crew

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_UNIQUE_PREVIEW_LIMIT = 10
_MAX_PROMPT_BYTES = 50_000


class AnalysisExplainer:
    """
    Data Analysis Pipeline.
    Loads a dataset from MongoDB GridFS, computes descriptive statistics using pandas, 
    and generates structured JSON insights via a 4-agent CrewAI pipeline.
    """

    def __init__(self, filename: str) -> None:
        try:
            self.filename = filename
            self.df = self._load_dataframe()

            logging.info("AnalysisExplainer initialised for file: %s", filename)
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def _require_env(key: str) -> str:
        """Return env variable *key* or raise a clear error if missing."""
        import os
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(f"Required environment variable '{key}' is not set.")
        return value

    def _load_dataframe(self):
        """Load dataset from MongoDB GridFS via utility helper."""
        try:
            df = load_dataframe_from_mongo(self.filename)
            logging.info("DataFrame loaded: %d rows × %d columns.", *df.shape)
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def _build_prompt_data(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Trim the full analysis dict down to what the crew actually needs."""
        prompt_data = {
            k: analysis[k]
            for k in ["shape", "columns", "null_values", "null_percentages"]
            if k in analysis
        }
        if "describe" in analysis:
            prompt_data["describe_sample"] = {
                c: {
                    s: round(v, 2) if isinstance(v, float) else v
                    for s, v in analysis["describe"][c].items()
                    if s in ("count", "mean", "std")
                }
                for c in list(analysis["describe"].keys())[:5]
            }
        if "duplicate_rows" in analysis:
            prompt_data["duplicate_rows"] = analysis["duplicate_rows"]

        if len(json.dumps(prompt_data, default=str).encode("utf-8")) > _MAX_PROMPT_BYTES:
            prompt_data = {
                "shape": analysis.get("shape"),
                "null_info": {k: v for k, v in analysis.get("null_percentages", {}).items() if v > 0},
                "duplicate_rows": analysis.get("duplicate_rows", 0),
            }
        return prompt_data

    def compute_analysis(self) -> dict[str, Any]:
        """Compute descriptive statistics using pandas."""
        try:
            df = self.df
            nulls = df.isnull().sum()

            return {
                "shape": df.shape,
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
                "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
                "null_values": nulls.to_dict(),
                "null_percentages": {c: round(v / len(df) * 100, 2) for c, v in nulls.items()},
                "duplicate_rows": int(df.duplicated().sum()),
                "describe": df.describe(include="all").to_dict(),
                "unique_counts": df.nunique().to_dict(),
                "sample_rows": df.head(5).to_dict(orient="records"),
            }
        except Exception as e:
            raise CustomException(e, sys) from e

    def unique_preview(self, limit: int = _UNIQUE_PREVIEW_LIMIT) -> dict[str, dict[str, Any]]:
        """Return a preview of unique values for every column."""
        try:
            return {
                col: {
                    "values": (u := self.df[col].dropna().unique())[:limit].tolist(),
                    "total_unique": len(u),
                    "truncated": len(u) > limit,
                }
                for col in self.df.columns
            }
        except Exception as e:
            raise CustomException(e, sys) from e

    def explain_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """
        Send pre-computed stats through the CrewAI pipeline to get structured insights.
        """
        prompt_data = self._build_prompt_data(analysis)

        for attempt in range(1, 4):
            try:
                crew = build_analysis_crew(prompt_data)
                result = crew.kickoff()

                insights = getattr(result, "pydantic", None)
                if insights is not None:
                    return self._validate_insights(insights.model_dump())

                logging.warning("Crew result had no structured pydantic output. Using stats fallback.")
                return self._build_insights_from_stats(analysis)

            except Exception as e:
                logging.warning("Error in explain_analysis attempt %d: %s", attempt, e)
                if attempt == 3:
                    return self._build_insights_from_stats(analysis)
                time.sleep(5 * attempt)

        return self._build_insights_from_stats(analysis)

    def _build_insights_from_stats(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Builds fallback insights when the crew/LLM analysis fails."""
        try:
            nulls = analysis.get("null_percentages", {})
            dups = analysis.get("duplicate_rows", 0)

            summary = f"Dataset has {analysis.get('shape', (0, 0))[0]} rows. "
            if dups: summary += f"Found {dups} duplicates. "

            flags = [
                {"column": c, "severity": "high" if p > 50 else "medium", "issue": "Missing values", "detail": f"{p}% missing"}
                for c, p in nulls.items() if p > 10
            ]
            if dups: flags.append({"column": "dataset", "severity": "medium", "issue": "Duplicate rows", "detail": f"Found {dups} duplicates."})

            insights = [{"column": col, "insight": f"Mean: {stats.get('mean', 0):.2f}, Std: {stats.get('std', 0):.2f}"}
                        for col in analysis.get("numeric_columns", [])[:5] 
                        if (stats := analysis.get("describe", {}).get(col, {}))]

            return {
                "summary": summary.strip(),
                "quality_flags": flags,
                "column_insights": insights,
                "next_steps": [{"title": "Review Data Quality", "detail": "Examine nulls and duplicates."}],
                "uncertainty_notes": "AI unavailable; based on raw stats.",
            }
        except Exception:
            return self._default_insights()

    def _validate_insights(self, insights: dict[str, Any]) -> dict[str, Any]:
        """Ensures all required keys exist and have correct types."""
        i = insights
        return {
            "summary": str(i.get("summary", "Dataset analysis completed.")),
            "quality_flags": [f for f in i.get("quality_flags", []) if isinstance(f, dict)],
            "column_insights": [c for c in i.get("column_insights", []) if isinstance(c, dict)],
            "next_steps": [s for s in i.get("next_steps", []) if isinstance(s, dict)],
            "uncertainty_notes": str(i.get("uncertainty_notes", "Standard statistical limitations apply.")),
        }

    def _default_insights(self) -> dict[str, Any]:
        """Return a valid default response."""
        return self._validate_insights({})

    def run(self) -> dict[str, Any]:
        """Execute the full pipeline: load → analyse → explain."""
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