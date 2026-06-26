from flask import Blueprint, request, jsonify

from services.auth_service import signup, login
from services.auth_decorator import require_auth


auth_bp = Blueprint(
    "auth",
    __name__
)


@auth_bp.route("/api/me", methods=["GET"])
@require_auth
def me_route():
    """Confirm the caller's JWT is valid. 401 (via require_auth) if not.

    The frontend hits this to verify a token server-side before granting access,
    so a stale/forged token in localStorage can't reach protected pages.
    """
    return jsonify({"email": request.user.get("email")}), 200



@auth_bp.route(
    "/auth/signup",
    methods=["POST"]
)
def signup_route():

    try:

        data = request.json

        email = data.get("email")
        password = data.get("password")
        confirm_password = data.get("confirm_password")

        result = signup(
            email,
            password,
            confirm_password
        )

        return jsonify({

            "token": result["token"],
            "email": result["email"]

        }), 200



    except Exception as e:

        message = e.args[0] if e.args else str(e)

        return jsonify({

            "detail": str(message)

        }), 400





@auth_bp.route(
    "/auth/login",
    methods=["POST"]
)
def login_route():

    try:

        data = request.json

        email = data.get("email")
        password = data.get("password")

        result = login(
            email,
            password
        )

        return jsonify({

            "token": result["token"],
            "email": result["email"]

        }), 200



    except Exception as e:

        message = e.args[0] if e.args else str(e)

        return jsonify({

            "detail": str(message)

        }), 401