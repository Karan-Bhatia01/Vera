import os
import json
import threading
import time
from flask import Blueprint, request, jsonify
from services.auth_decorator import require_auth
from src.components.data_info import DataInfo
from src.components.mongo_storage import store_dataset_insights
from src.components.job_store import create_job, update_job, get_job_status
from src.components.llm import AnalysisExplainer
from src.logger import logging

analysis_bp = Blueprint("analysis", __name__)


def _run_info_analysis(job_id: str, filename: str):
    t0 = time.time()
    try:
        update_job(job_id, status="running", progress=10, message="Computing stats...")
        analyzer = DataInfo(filename)
        analysis = analyzer.dataset_analysis()
        unique = analyzer.get_unique_column_values()

        update_job(job_id, progress=60, message="Generating insights using CrewAI...")
        explainer = AnalysisExplainer(filename)
        ai_insights = explainer.explain_analysis(analysis)

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