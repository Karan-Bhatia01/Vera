import os
import threading
from typing import Any

from dotenv import load_dotenv

# Load env once for the whole app, forcing .env to override system vars
load_dotenv(override=True)

from pymongo import MongoClient
import gridfs
import openai
from groq import Groq
from langchain_groq import ChatGroq

# ──────────────────────────────────────────────
# Global singletons
# ──────────────────────────────────────────────
_mongo_client: MongoClient | None = None
_mongo_db = None
_gridfs = None

_groq_client: Groq | None = None
_chatgroq_client: ChatGroq | None = None
_openai_vision_client: openai.OpenAI | None = None

# ──────────────────────────────────────────────
# Threading locks for thread-safe lazy init
# ──────────────────────────────────────────────
_mongo_lock = threading.RLock()
_groq_lock = threading.RLock()
_chatgroq_lock = threading.RLock()
_openai_lock = threading.RLock()


def get_mongo_client() -> MongoClient:
    global _mongo_client
    with _mongo_lock:
        if _mongo_client is None:
            mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
            _mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
    return _mongo_client


def get_db():
    global _mongo_db
    with _mongo_lock:
        if _mongo_db is None:
            client = get_mongo_client()
            db_name = os.environ.get("MONGO_DB", "clarityAI_database")
            _mongo_db = client[db_name]
    return _mongo_db


def get_gridfs():
    global _gridfs
    with _mongo_lock:
        if _gridfs is None:
            db = get_db()
            _gridfs = gridfs.GridFS(db)
    return _gridfs


def get_groq_client(timeout: float = 45.0) -> Groq:
    global _groq_client
    with _groq_lock:
        if _groq_client is None:
            # We ignore dynamic timeout overrides if already initialized,
            # as it's better to share the connection pool.
            _groq_client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
                timeout=timeout,
                max_retries=2
            )
    return _groq_client


def get_chatgroq_client() -> ChatGroq:
    global _chatgroq_client
    with _chatgroq_lock:
        if _chatgroq_client is None:
            model = os.environ.get("GROQ_LLM_MODEL", "groq/openai/gpt-oss-120b")
            _chatgroq_client = ChatGroq(
                api_key=os.environ.get("GROQ_API_KEY"),
                model=model,
                temperature=0.3,
                max_tokens=4096,
            )
    return _chatgroq_client

def get_gemini_client() -> str:
    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" in os.environ:
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    model = os.environ.get("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
    return f"gemini/{model}"



def get_openai_vision_client() -> openai.OpenAI:
    global _openai_vision_client
    with _openai_lock:
        if _openai_vision_client is None:
            base_url = os.environ.get("GROQ_OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
            api_key = os.environ.get("GROQ_API_KEY")
            _openai_vision_client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=90,
            )
    return _openai_vision_client
