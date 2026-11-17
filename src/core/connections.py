import os
import threading

from dotenv import load_dotenv

# Load env once for the whole app, forcing .env to override system vars
load_dotenv(override=True)

from pymongo import MongoClient
import gridfs
from groq import Groq

# Lazily created, thread-safe singletons shared by the whole app.
_mongo_client: MongoClient | None = None
_mongo_db = None
_gridfs = None
_groq_client: Groq | None = None

_mongo_lock = threading.RLock()
_groq_lock = threading.RLock()


def get_mongo_client() -> MongoClient:
    global _mongo_client
    with _mongo_lock:
        if _mongo_client is None:
            mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
            _mongo_client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=10000,
                tls=("mongodb+srv" in mongo_uri),
                retryWrites=True,
            )
    return _mongo_client


def get_db():
    global _mongo_db
    with _mongo_lock:
        if _mongo_db is None:
            db_name = os.environ.get("DB_NAME", "clarityAI_database")
            _mongo_db = get_mongo_client()[db_name]
    return _mongo_db


def get_gridfs():
    global _gridfs
    with _mongo_lock:
        if _gridfs is None:
            _gridfs = gridfs.GridFS(get_db())
    return _gridfs


def get_groq_client(timeout: float = 45.0) -> Groq:
    global _groq_client
    with _groq_lock:
        if _groq_client is None:
            # The first caller's timeout sticks; per-request timeouts are passed on each call.
            _groq_client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
                timeout=timeout,
                max_retries=2,
            )
    return _groq_client
