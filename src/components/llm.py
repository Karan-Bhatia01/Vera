from __future__ import annotations

import sys
import json
from typing import Any

from pydantic import BaseModel, Field

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo, parse_json_response
from src.agents.llm_provider import chat_completion

_MODEL = "openai/gpt-oss-120b"
_TIMEOUT = 120  # long JSON replies (10 charts with takeaways) need more than the default
_MAX_PROMPT_BYTES = 20_000
_MAX_TURNS = 10


# ── Response schemas ─────────────────────────────────────────────────────────

class QualityFlag(BaseModel):
    column: str
    severity: str = Field(description="high, medium, or low")
    issue: str
    detail: str

class ColumnInsight(BaseModel):
    column: str
    insight: str

class TitledDetail(BaseModel):
    title: str
    detail: str

class RecommendedTarget(BaseModel):
    column: str
    reason: str

class AICorrelation(BaseModel):
    columns: list[str]
    detail: str

class RecommendedChart(BaseModel):
    chart_type: str = Field(description="'bar', 'line', 'scatter', 'pie', 'doughnut', 'histogram', 'radar', 'polarArea', or 'bubble'")
    columns: list[str] = Field(description="Columns involved (1 for pie, 2 for scatter, 3 for bubble).")
    story_takeaway: str = Field(description="At least 150 words: 3 detailed paragraphs with bullet points on the business, statistical and strategic meaning of this chart.")

class DatasetInsights(BaseModel):
    summary: str
    business_insights: str = Field(description="A story-telling evaluation of the dataset's business value and a final judgement.")
    recommended_charts: list[RecommendedChart]
    recommended_target: RecommendedTarget
    next_steps: list[TitledDetail]
    quality_flags: list[QualityFlag]
    preprocessing: list[TitledDetail]
    uncertainty_notes: str
    feature_engineering: list[TitledDetail]
    column_insights: list[ColumnInsight]
    correlations: list[AICorrelation]

class MLInsight(BaseModel):
    executive_summary: str = Field(description="1-2 sentences summarizing the model's overall performance in simple terms.")
    business_impact: str = Field(description="How this model can be used in a real business context, translated from the metrics.")
    top_features: list[str] = Field(description="The most important features driving the predictions, in plain English.")
    confidence_level: str = Field(description="High, Medium, or Low confidence in the model, based on its scores.")
    simple_explanation: str = Field(description="What the best model is doing under the hood, using analogies instead of math.")


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an elite Senior Data Scientist.
You will be provided with statistical analysis of a dataset.
Your job is to read these stats, use your tools to investigate further if needed, and generate structured insights.

Look for:
- High cardinality (IDs) or constant columns.
- Skewness or high outlier counts.
- High null percentages.
- Interesting correlations.
Provide an executive summary, flag data quality issues, provide column-level insights, suggest next steps, AND critically, provide a structured story-telling evaluation of the dataset's business value under 'business_insights'.
ALSO, curate a list of 'recommended_charts'.
CRITICAL MANDATORY INSTRUCTION: You MUST generate EXACTLY 10 charts. Do not generate 5, do not generate 9, generate EXACTLY 10.
You MUST include a mix of:
- Univariate (1 column): histogram, pie, doughnut, polarArea
- Bivariate (2 columns): bar, line, scatter, radar
- Trivariate/Multivariate (3 columns): bubble
CRITICAL: The 'story_takeaway' for each chart MUST be extremely detailed (at least 150 words, multiple paragraphs, and bullet points). Do NOT just write one line.

IMPORTANT: You MUST return a strictly valid JSON object matching this schema.
DO NOT wrap the output in a tool call. DO NOT attempt to call a tool named 'JSON'. Just output the raw JSON directly as your reply text.
CRITICAL: Do NOT use markdown formatting (like **bold** or `code`) INSIDE the JSON string values. The response MUST be perfectly parseable by `json.loads()`.
{
    "summary": "string",
    "business_insights": "string",
    "recommended_charts": [{"chart_type": "bar|line|scatter|pie|doughnut|histogram|radar|polarArea|bubble", "columns": ["col1", "col2"], "story_takeaway": "string"}],
    "recommended_target": {"column": "string", "reason": "string"},
    "next_steps": [{"title": "string", "detail": "string"}],
    "quality_flags": [{"column": "string", "severity": "high|medium|low", "issue": "string", "detail": "string"}],
    "preprocessing": [{"title": "string", "detail": "string"}],
    "uncertainty_notes": "string",
    "feature_engineering": [{"title": "string", "detail": "string"}],
    "column_insights": [{"column": "string", "insight": "string"}],
    "correlations": [{"columns": ["col1", "col2"], "detail": "string"}]
}
"""

ML_PROMPT = """You are an elite Senior Data Scientist reviewing machine learning results.
I will give you the results of an automated Machine Learning pipeline.
Translate these metrics, models, and feature importances into a simple, structured business analysis that a non-technical manager can understand.
Do not use heavy math jargon. Focus on WHAT is predicting the target, HOW accurate it is, and WHY it matters.

Here are the ML results:
{results}

IMPORTANT: You MUST return a strictly valid JSON object matching this schema.
DO NOT wrap the output in a tool call. Just output the raw JSON directly as your reply text.
{{
    "executive_summary": "string",
    "business_impact": "string",
    "top_features": ["string"],
    "confidence_level": "string",
    "simple_explanation": "string"
}}
"""

# The model may peek at value counts before answering. It sometimes also invents
# a "JSON" tool to hand back its answer, so we accept that as the final reply.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_value_distribution",
            "description": "Get exact value counts for a categorical column.",
            "parameters": {
                "type": "object",
                "properties": {"column_name": {"type": "string"}},
                "required": ["column_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "JSON",
            "description": "Use this to output the final JSON summary if you decide to use a tool call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "business_insights": {"type": "string"},
                    "recommended_charts": {"type": "array"},
                    "recommended_target": {"type": "object"},
                    "next_steps": {"type": "array"},
                    "quality_flags": {"type": "array"},
                    "preprocessing": {"type": "array"},
                    "uncertainty_notes": {"type": "string"},
                    "feature_engineering": {"type": "array"},
                    "column_insights": {"type": "array"},
                    "correlations": {"type": "array"},
                },
            },
        },
    },
]


def _parse(schema: type[BaseModel], text: str) -> dict:
    """Validate an LLM reply against a schema. Raises if it doesn't fit."""
    return schema.model_validate(parse_json_response(text or "")).model_dump()


def _ask_to_fix(messages: list, bad_reply: str, error: Exception) -> None:
    """Feed a bad reply back to the model so it can correct its JSON."""
    messages.append({"role": "assistant", "content": bad_reply})
    messages.append({
        "role": "user",
        "content": f"Your last response failed JSON validation with this error:\n{error}\n\n"
                   "Please fix the formatting and return ONLY the valid JSON object matching the schema.",
    })


# ── Dataset insights ─────────────────────────────────────────────────────────

class AnalysisExplainer:
    """Turns pre-computed dataset stats into structured AI insights (one Groq chat with tools)."""

    def __init__(self, filename: str, df=None) -> None:
        try:
            self.filename = filename
            # Reuse an already-loaded frame when the caller has one, to skip a GridFS download.
            self.df = df if df is not None else load_dataframe_from_mongo(filename)
            logging.info("AnalysisExplainer initialised for file: %s", filename)
        except Exception as e:
            raise CustomException(e, sys) from e

    def query_value_distribution(self, column_name: str) -> str:
        """Exact value counts for one column (tool the model can call)."""
        if column_name not in self.df.columns:
            return f"Error: Column {column_name} not found."
        return self.df[column_name].value_counts(dropna=False).head(20).to_json()

    def _build_prompt_data(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Trim the full analysis dict down to stay within token limits."""
        keys = ["shape", "columns", "null_percentages", "advanced_numeric",
                "advanced_categorical", "cardinality_flags", "correlations"]
        prompt_data = {k: analysis[k] for k in keys if k in analysis}

        if len(json.dumps(prompt_data, default=str).encode("utf-8")) > _MAX_PROMPT_BYTES:
            prompt_data = {
                "shape": analysis.get("shape"),
                "cardinality_flags": analysis.get("cardinality_flags"),
                "null_info": {k: v for k, v in analysis.get("null_percentages", {}).items() if v > 0},
            }
        return prompt_data

    def explain_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Ask the model for insights, letting it call tools and fix bad JSON along the way."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here are the dataset statistics:\n{json.dumps(self._build_prompt_data(analysis), default=str)}"},
        ]

        try:
            for _ in range(_MAX_TURNS):
                message = chat_completion(
                    messages, model=_MODEL, timeout=_TIMEOUT,
                    tools=TOOLS, temperature=0.4, max_tokens=8000,
                )

                if message.tool_calls:
                    messages.append(message.model_dump(exclude_unset=True))
                    for call in message.tool_calls:
                        args = json.loads(call.function.arguments)
                        if call.function.name.upper() == "JSON":
                            return DatasetInsights.model_validate(args).model_dump()

                        if call.function.name == "query_value_distribution":
                            result = self.query_value_distribution(args.get("column_name", ""))
                        else:
                            result = "Unknown tool."
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                    continue

                if not message.content:
                    break

                try:
                    return _parse(DatasetInsights, message.content)
                except Exception as error:
                    logging.warning("AI JSON parse failed, asking it to retry. Error: %s", error)
                    _ask_to_fix(messages, message.content, error)

        except Exception as e:
            logging.warning("Error in explain_analysis: %s", e)

        return self._fallback_insights()

    @staticmethod
    def _fallback_insights() -> dict[str, Any]:
        """Empty-but-valid insights so the UI still renders when the AI fails."""
        return {
            "summary": "AI generation failed. Fallback stats provided.",
            "business_insights": "Business insights are unavailable because AI generation failed.",
            "recommended_charts": [],
            "recommended_target": {"column": "None", "reason": "AI unavailable."},
            "next_steps": [{"title": "Retry", "detail": "Try running the analysis again."}],
            "quality_flags": [],
            "preprocessing": [],
            "uncertainty_notes": "AI unavailable.",
            "feature_engineering": [],
            "column_insights": [],
            "correlations": [],
        }


# ── ML results explainer ─────────────────────────────────────────────────────

def explain_ml_metrics(ml_result: dict) -> dict:
    """Translate raw ML metrics into a plain-English business summary."""
    prompt = ML_PROMPT.format(results=json.dumps(ml_result, default=str)[:15000])
    messages = [{"role": "user", "content": prompt}]
    error: Exception | None = None

    for _ in range(3):
        reply = ""
        try:
            reply = chat_completion(
                messages, model=_MODEL, timeout=_TIMEOUT,
                temperature=0.3, max_tokens=2000,
            ).content
            return _parse(MLInsight, reply)
        except Exception as e:
            error = e
            logging.warning("explain_ml_metrics failed, asking AI to retry. Error: %s", e)
            _ask_to_fix(messages, reply, e)

    return {
        "executive_summary": "Failed to generate AI analysis.",
        "business_impact": "N/A",
        "top_features": [],
        "confidence_level": "Unknown",
        "simple_explanation": str(error),
    }
