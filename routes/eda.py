import threading
from flask import Blueprint, request, jsonify
from services.auth_decorator import require_auth
from src.components.eda_processing import DataPreprocessing
from src.utils import analyse_chart
from src.components.job_store import create_job, update_job, get_job_status
from src.agents.missing_value_agent import decide_missing_value_strategy
from src.logger import logging

eda_bp = Blueprint("eda", __name__)


def _run_eda(job_id: str, filename: str, target_column: str, columns_to_drop: list):
    try:
        update_job(job_id, status="running", message="Preprocessing data...")
        processor = DataPreprocessing(
            filename=filename,
            target_column=target_column,
            columns_to_drop=columns_to_drop,
        )
        strategy = decide_missing_value_strategy(processor.df)
        update_job(job_id, message="Generating charts...")
        cleaned_df = processor.preprocess_data(strategy)
        charts = processor.generate_eda_report(cleaned_df)
        update_job(job_id, status="completed", result={
            "charts": charts,
            "columns": cleaned_df.columns.tolist(),
            "shape": list(cleaned_df.shape),
        })
    except Exception as e:
        logging.exception("EDA job %s failed", job_id)
        update_job(job_id, status="failed", error=str(e))


@eda_bp.route("/api/run_eda", methods=["POST"])
@require_auth
def run_eda():
    try:
        data = request.json
        filename = data.get("filename")
        target_column = data.get("target_column")
        if not filename or not target_column:
            return jsonify({"status": "error", "message": "filename and target_column required"}), 400
        job_id = create_job()
        thread = threading.Thread(
            target=_run_eda,
            args=(job_id, filename, target_column, data.get("columns_to_drop", [])),
            daemon=True,
        )
        thread.start()
        return jsonify({"status": "success", "eda_job_id": job_id}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@eda_bp.route("/api/pipeline_status/eda/<job_id>", methods=["GET"])
@require_auth
def eda_status(job_id):
    result = get_job_status(job_id)
    if result is None:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    return jsonify(result), 200


@eda_bp.route("/api/analyse_chart", methods=["POST"])
@require_auth
def analyse_chart_route():
    try:
        data = request.json
        result = analyse_chart(
            data.get("image_b64"),
            data.get("chart_title"),
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500