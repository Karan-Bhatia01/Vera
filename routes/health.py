from flask import Blueprint, jsonify

from src.components.mongo_storage import get_db


health_bp = Blueprint(
    "health",
    __name__
)


@health_bp.route("/api/health", methods=["GET"])
def health_check():

    # Actually ping MongoDB so the frontend can show a live "connected"
    # indicator. A failed/unreachable Mongo raises, which we report as
    # "disconnected" instead of letting the request hang or 500.
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