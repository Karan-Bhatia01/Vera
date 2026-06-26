from flask import Blueprint, request, jsonify

from services.upload_service import handle_upload
from services.auth_decorator import require_auth


upload_bp = Blueprint(
    "upload",
    __name__
)


@upload_bp.route(
    "/api/upload",
    methods=["POST"]
)
@require_auth
def upload_file():

    file = request.files.get("file")

    if not file:

        return jsonify({
            "status": "error",
            "message": "No file uploaded"
        }), 400

    try:

        owner_email = request.user["email"]

        result = handle_upload(file, owner_email)

        return jsonify({

            "status": "success",
            "message": "File uploaded successfully",
            "data": result

        }), 200

    except ValueError as e:

        return jsonify({

            "status": "error",
            "message": str(e)

        }), 400

    except Exception as e:

        message = e.args[0] if e.args else str(e)

        return jsonify({

            "status": "error",
            "message": str(message)

        }), 500