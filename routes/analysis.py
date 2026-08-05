import os
import json
import threading
import time
from flask import Blueprint, request, jsonify
from services.auth_decorator import require_auth
from src.components.data_info import DataInfo
from src.components.mongo_storage import store_dataset_insights
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

def _default_insights() -> dict:
    return {
        "summary": "Analysis completed.", "quality_flags": [], "column_insights": [],
        "correlations": [], "recommended_target": {}, "feature_engineering": [],
        "preprocessing": [], "next_steps": [], "uncertainty_notes": ""
    }

def _build_ai_insights(analysis: dict) -> dict:
    """Single LLM call to generate insights from pre-computed stats."""
    try:
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
            "numeric_stats": dict(list(analysis.get("numeric_stats", {}).items())[:10]),
            "correlations": analysis.get("correlations", [])[:8],
        }
        if top_cats := analysis.get("top_categories", {}):
            payload["top_categories"] = {c: v[:5] for c, v in list(top_cats.items())[:8]}

        raw = call_llm(
            _SYSTEM_PROMPT, json.dumps(payload, default=str), temperature=0.4,
            model=os.environ.get("GROQ_INSIGHTS_MODEL", "llama-3.3-70b-versatile"),
            timeout=float(os.environ.get("GROQ_INSIGHTS_TIMEOUT", "45")),
            max_tokens=int(os.environ.get("GROQ_INSIGHTS_MAX_TOKENS", "3000")),
        )
        
        res = parse_json_from_response(raw)
        if not isinstance(res, dict): return _default_insights()

        return {
            "summary": res.get("summary", ""),
            "quality_flags": res.get("quality_flags", []),
            "column_insights": res.get("column_insights", []),
            "correlations": res.get("correlations", []),
            "recommended_target": res.get("recommended_target", {}),
            "feature_engineering": res.get("feature_engineering", []),
            "preprocessing": res.get("preprocessing", []),
            "next_steps": res.get("next_steps", []),
            "uncertainty_notes": res.get("uncertainty_notes", ""),
        }
    except Exception:
        return _default_insights()

def _run_info_analysis(job_id: str, filename: str):
    t0 = time.time()
    try:
        update_job(job_id, status="running", progress=10, message="Computing stats...")
        analyzer = DataInfo(filename)
        analysis = analyzer.dataset_analysis()
        unique = analyzer.get_unique_column_values()

        update_job(job_id, progress=60, message="Generating insights...")
        ai_insights = _build_ai_insights(analysis)

        store_dataset_insights(filename, analysis, ai_insights, unique)
        result = {"analysis": analysis, "unique": unique, "ai_insights": ai_insights}
        
        update_job(job_id, status="completed", progress=100, message="Done", result=result)
        logging.info("info job %s: completed in %.1fs", job_id, time.time() - t0)
    except Exception as e:
        logging.exception("info job %s failed", job_id)
        update_job(job_id, status="failed", error=str(e))


@analysis_bp.route("/api/info/<filename>", methods=["GET"])
@require_auth
def dataset_info(filename):
    try:
        analyzer = DataInfo(filename)
        return jsonify({"status": "success", "data": {
            **analyzer.dataset_analysis(), "unique_values": analyzer.get_unique_column_values()
        }}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@analysis_bp.route("/api/run_info", methods=["POST"])
@require_auth
def run_info():
    try:
        if not (filename := request.json.get("filename")):
            return jsonify({"status": "error", "message": "filename required"}), 400

        job_id = create_job()
        threading.Thread(target=_run_info_analysis, args=(job_id, filename), daemon=True).start()
        return jsonify({"status": "success", "info_job_id": job_id}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@analysis_bp.route("/api/pipeline_status/info/<job_id>", methods=["GET"])
@require_auth
def info_status(job_id):
    if result := get_job_status(job_id):
        return jsonify(result), 200
    return jsonify({"status": "error", "message": "Job not found"}), 404