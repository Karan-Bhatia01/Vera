import os
import json
import threading
import time
from flask import Blueprint, request, jsonify
from services.auth_decorator import require_auth
from src.components.data_info import DataInfo
from src.components.mongo_storage import get_dataset_insights, store_dataset_insights
from src.components.job_store import create_job, update_job, get_job_status
from src.agents.llm_provider import call_llm, parse_json_from_response
from src.logger import logging

analysis_bp = Blueprint("analysis", __name__)

_SYSTEM_PROMPT = """\
You are a senior data analyst. Study the dataset summary and return ONLY valid JSON with these EXACT keys:
{
  "summary": "<4-6 sentences: infer the likely domain/purpose of the dataset, describe its structure, the kinds of columns it holds, and any standout characteristics>",
  "quality_flags": [{"column":"","severity":"high|medium|low","issue":"<short label>","detail":"<specific, actionable explanation referencing the numbers>"}],
  "column_insights": [{"column":"","insight":"<concrete observation, cite ranges/means/cardinality where available>"}],
  "correlations": [{"columns":["colA","colB"],"detail":"<what this relationship implies and whether it risks leakage/multicollinearity>"}],
  "recommended_target": {"column":"<best column to predict>","reason":"<why it's a plausible ML target>"},
  "feature_engineering": [{"title":"","detail":"<a specific transformation/derived feature and why>"}],
  "preprocessing": [{"title":"","detail":"<encoding/scaling/imputation/outlier step tied to this data>"}],
  "next_steps": [{"title":"","detail":""}],
  "uncertainty_notes": "<what you could not determine and why>"
}
Be thorough and specific to THIS dataset. Provide column_insights for at least 5 meaningful columns, flag every genuine quality concern, and make recommendations concrete (name columns). Do NOT invent values that aren't supported by the summary. No markdown, no prose outside the JSON.
"""

def _build_ai_insights(analysis: dict) -> dict:
    """Single LLM call to generate insights from pre-computed stats."""
    try:
        # Compact payload — enough context for detailed insights, still small.
        payload = {
            "shape": analysis.get("shape"),
            "columns": analysis.get("columns", [])[:20],
            "dtypes": analysis.get("dtypes", {}),
            "null_percentages": {k: v for k, v in analysis.get("null_percentages", {}).items() if v > 0},
            "duplicate_rows": analysis.get("duplicate_rows"),
            "memory_usage_mb": analysis.get("memory_usage_mb"),
            "numeric_columns": analysis.get("numeric_columns", []),
            "categorical_columns": analysis.get("categorical_columns", []),
            "unique_counts": analysis.get("unique_counts", {}),
            # Pre-computed stats so the LLM reasons over real numbers, not guesses.
            "numeric_stats": dict(list(analysis.get("numeric_stats", {}).items())[:10]),
            "correlations": analysis.get("correlations", [])[:8],
        }
        # Top categories, capped so the prompt stays compact.
        top_cats = analysis.get("top_categories", {})
        if top_cats:
            payload["top_categories"] = {
                col: vals[:5] for col, vals in list(top_cats.items())[:8]
            }

        # Plain JSON generation — use a fast chat model with a tight timeout,
        # not the slow agentic compound model. Falls back to defaults on timeout.
        raw = call_llm(
            _SYSTEM_PROMPT,
            json.dumps(payload, default=str),
            temperature=0.4,
            model=os.environ.get("GROQ_INSIGHTS_MODEL", "llama-3.3-70b-versatile"),
            timeout=float(os.environ.get("GROQ_INSIGHTS_TIMEOUT", "45")),
            # The insights JSON has many sections; a small cap truncates it
            # mid-object and the whole response fails to parse.
            max_tokens=int(os.environ.get("GROQ_INSIGHTS_MAX_TOKENS", "3000")),
        )
        result = parse_json_from_response(raw)
        if not isinstance(result, dict):
            return _default_insights()
        # Ensure every expected key exists with the right empty shape so the
        # frontend can render unconditionally even if the model omits some.
        _STR_KEYS  = {"summary", "uncertainty_notes"}
        _DICT_KEYS = {"recommended_target"}
        _LIST_KEYS = {
            "quality_flags", "column_insights", "correlations",
            "feature_engineering", "preprocessing", "next_steps",
        }
        for k in _STR_KEYS:
            result.setdefault(k, "")
        for k in _DICT_KEYS:
            result.setdefault(k, {})
        for k in _LIST_KEYS:
            result.setdefault(k, [])
        return result
    except Exception:
        return _default_insights()

def _insights_are_current(existing: dict) -> bool:
    """True only if a cached Mongo doc has the current analysis/AI shape.

    Bump the marker keys here whenever the stored shape changes so older
    cached documents auto-invalidate and get recomputed on next request.
    """
    analysis = existing.get("analysis") or {}
    ai = existing.get("ai_insights") or {}
    return "numeric_stats" in analysis and "recommended_target" in ai


def _default_insights():
    return {
        "summary": "Analysis completed.",
        "quality_flags": [],
        "column_insights": [],
        "correlations": [],
        "recommended_target": {},
        "feature_engineering": [],
        "preprocessing": [],
        "next_steps": [],
        "uncertainty_notes": "",
    }

def _run_info_analysis(job_id: str, filename: str):
    t0 = time.time()
    try:
        update_job(job_id, status="running", progress=10, message="Computing dataset stats...")

        # Step 1: compute stats (pure Python, no LLM)
        from src.components.data_info import DataInfo
        analyzer = DataInfo(filename)
        analysis = analyzer.dataset_analysis()
        unique = analyzer.get_unique_column_values()
        logging.info("info job %s: stats computed in %.1fs", job_id, time.time() - t0)

        update_job(job_id, progress=60, message="Generating AI insights...")

        # Step 2: ONE LLM call (time-bounded; degrades to default insights)
        t_llm = time.time()
        ai_insights = _build_ai_insights(analysis)
        logging.info("info job %s: insights in %.1fs", job_id, time.time() - t_llm)

        # Step 3: store to Mongo
        store_dataset_insights(filename, analysis, ai_insights, unique)

        result = {
            "analysis": analysis,
            "unique": unique,
            "ai_insights": ai_insights,
        }
        update_job(job_id, status="completed", progress=100, message="Done", result=result)
        logging.info("info job %s: completed in %.1fs", job_id, time.time() - t0)

    except Exception as e:
        logging.exception("info job %s failed after %.1fs", job_id, time.time() - t0)
        update_job(job_id, status="failed", error=str(e))


@analysis_bp.route("/api/info/<filename>", methods=["GET"])
@require_auth
def dataset_info(filename):
    try:
        analyzer = DataInfo(filename)
        result = analyzer.dataset_analysis()
        result["unique_values"] = analyzer.get_unique_column_values()
        return jsonify({"status": "success", "data": result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@analysis_bp.route("/api/run_info", methods=["POST"])
@require_auth
def run_info():
    try:
        filename = request.json.get("filename")
        if not filename:
            return jsonify({"status": "error", "message": "filename required"}), 400

        job_id = create_job()
        thread = threading.Thread(
            target=_run_info_analysis, args=(job_id, filename), daemon=True
        )
        thread.start()
        return jsonify({"status": "success", "info_job_id": job_id}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@analysis_bp.route("/api/pipeline_status/info/<job_id>", methods=["GET"])
@require_auth
def info_status(job_id):
    result = get_job_status(job_id)
    if result is None:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    return jsonify(result), 200