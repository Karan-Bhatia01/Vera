from functools import wraps

from flask import request, jsonify

from services.auth_service import verify_token


def require_auth(f):
    """
    Protects a route by requiring a valid JWT in the Authorization header.

    Expected header format: Authorization: Bearer <token>

    On success, attaches the decoded payload to request.user (so route
    handlers can access request.user["email"] if needed) and calls
    the wrapped function normally.

    On failure, returns 401 with a clean error message — never calls
    the wrapped function.
    """

    @wraps(f)
    def decorated(*args, **kwargs):

        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):

            return jsonify({

                "detail": "Missing or malformed Authorization header"

            }), 401

        token = auth_header.split("Bearer ", 1)[1].strip()

        if not token:

            return jsonify({

                "detail": "Missing token"

            }), 401

        try:

            payload = verify_token(token)

        except Exception as e:

            message = e.args[0] if e.args else str(e)

            return jsonify({

                "detail": str(message)

            }), 401

        request.user = payload

        return f(*args, **kwargs)

    return decorated