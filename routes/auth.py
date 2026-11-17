from flask import Blueprint, request, jsonify

from services.auth_service import signup, login, google_auth
from services.auth_decorator import require_auth


auth_bp = Blueprint("auth", __name__)


def _auth_response(action, error_status):
    """Run an auth action and return its token, or a clean error message."""
    try:
        result = action(request.json or {})
        return jsonify({"token": result["token"], "email": result["email"]}), 200
    except Exception as e:
        message = e.args[0] if e.args else str(e)
        return jsonify({"detail": str(message)}), error_status


@auth_bp.route("/api/me", methods=["GET"])
@require_auth
def me_route():
    """Confirm the caller's JWT is valid. 401 (via require_auth) if not.

    The frontend hits this to verify a token server-side before granting access,
    so a stale/forged token in localStorage can't reach protected pages.
    """
    return jsonify({"email": request.user.get("email")}), 200


@auth_bp.route("/auth/signup", methods=["POST"])
def signup_route():
    return _auth_response(
        lambda data: signup(data.get("email"), data.get("password"), data.get("confirm_password")),
        error_status=400,
    )


@auth_bp.route("/auth/login", methods=["POST"])
def login_route():
    return _auth_response(
        lambda data: login(data.get("email"), data.get("password")),
        error_status=401,
    )


@auth_bp.route("/auth/google", methods=["POST"])
def google_auth_route():
    return _auth_response(
        lambda data: google_auth(data.get("token")),
        error_status=400,
    )
