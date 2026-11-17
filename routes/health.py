from flask import Blueprint, jsonify

from src.core.connections import get_db


health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health_check():
    # Ping MongoDB so the frontend can show a live "connected" badge.
    # An unreachable database is reported as "disconnected" instead of a 500.
    try:
        get_db().client.admin.command("ping")
        database = "connected"
    except Exception:
        database = "disconnected"

    return jsonify({
        "status": "success",
        "message": "Vera API is running",
        "database": database,
    }), 200
