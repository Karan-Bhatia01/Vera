from flask import Blueprint, request, jsonify
from services.auth_decorator import require_auth
from src.agents.chat_agent import chat as agent_chat

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    try:
        data = request.json
        query = data.get("query")
        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400

        result = agent_chat(
            query=query,
            filename=data.get("filename"),
            history=data.get("history", [])
        )
        return jsonify({"status": "success", "response": result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500