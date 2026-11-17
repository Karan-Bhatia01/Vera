import threading
from flask import Blueprint, request, jsonify
from services.auth_decorator import require_auth
from src.ml import MLPipeline
from src.components.job_store import create_job, update_job, get_job, get_job_status
from src.components.llm import explain_ml_metrics
from src.logger import logging

ml_bp = Blueprint("ml", __name__)


def _run_training(job_id: str, filename: str, target_column: str):
    try:
        update_job(job_id, status="running", message="Starting ML pipeline...")
        pipeline = MLPipeline(filename=filename, target_column=target_column)
        result = pipeline.run(progress_callback=lambda pct, msg: update_job(job_id, message=msg))
        update_job(job_id, status="completed", result=result)
    except Exception as e:
        logging.exception("ML job %s failed", job_id)
        update_job(job_id, status="failed", error=str(e))


@ml_bp.route("/api/run_ml", methods=["POST"])
@require_auth
def run_ml():
    try:
        data = request.json
        filename = data.get("filename")
        target_column = data.get("target_column")
        if not filename or not target_column:
            return jsonify({"status": "error", "message": "filename and target_column required"}), 400
        job_id = create_job()
        thread = threading.Thread(
            target=_run_training, args=(job_id, filename, target_column), daemon=True
        )
        thread.start()
        return jsonify({"status": "success", "ml_job_id": job_id}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@ml_bp.route("/api/pipeline_status/ml/<job_id>", methods=["GET"])
@require_auth
def ml_status(job_id):
    result = get_job_status(job_id)
    if result is None:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    return jsonify(result), 200


def _run_ml_analysis(job_id: str, ml_result: dict):
    try:
        update_job(job_id, status="running", message="Generating executive analysis...")
        update_job(job_id, status="completed", result=explain_ml_metrics(ml_result))
    except Exception as e:
        logging.exception("ML analysis job %s failed", job_id)
        update_job(job_id, status="failed", error=str(e))


@ml_bp.route("/api/analyze_ml/<ml_job_id>", methods=["POST"])
@require_auth
def analyze_ml(ml_job_id):
    """Start the plain-English LLM summary of a finished ML job in the background."""
    ml_job = get_job(ml_job_id)
    if not ml_job:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    if ml_job.get("status") != "completed" or not ml_job.get("result"):
        return jsonify({"status": "error", "message": "Job not completed yet"}), 400

    job_id = create_job()
    threading.Thread(target=_run_ml_analysis, args=(job_id, ml_job["result"]), daemon=True).start()
    return jsonify({"status": "success", "analysis_job_id": job_id}), 200


@ml_bp.route("/api/pipeline_status/analysis/<job_id>", methods=["GET"])
@require_auth
def ml_analysis_status(job_id):
    result = get_job_status(job_id)
    if result is None:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    return jsonify(result), 200
