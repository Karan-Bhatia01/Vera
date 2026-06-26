"""
Job store for tracking async pipeline jobs (info / eda / ml).

State lives in an in-memory dict mirrored to a JSON file on disk so that jobs
survive Flask reloads and restarts during development. Without this, a restart
wipes every job while the browser still holds the old ids in localStorage —
every poll then 404s forever.

Mongo remains the source of truth for finished insights; this store only tracks
transient job progress/results and is cleaned up after an hour.
"""

import json
import os
import threading
import time
import uuid

_TTL = 3600  # seconds; jobs older than this are purged
_lock = threading.RLock()

# Project-root/.runtime/jobs.json (this file lives at src/components/job_store.py)
_STORE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".runtime", "jobs.json")
)


def _load() -> dict:
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return {}


_jobs: dict[str, dict] = _load()


def _flush():
    """Persist current state atomically. Best-effort: in-memory is authoritative."""
    try:
        os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
        tmp = f"{_STORE_PATH}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_jobs, f, default=str)
        os.replace(tmp, _STORE_PATH)
    except OSError:
        pass


def create_job() -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Queued...",
            "result": None,
            "error": None,
            "created": time.time(),
        }
        _flush()
    return job_id


def update_job(job_id: str, **kwargs):
    """Update job state."""
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _flush()


def get_job(job_id: str) -> dict | None:
    """Get job state (returns a copy)."""
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def get_job_status(job_id: str) -> dict | None:
    """Get job state for status polling.

    While the job is still in progress, the heavy `result` payload is omitted so
    that 3s status polls stay small. The full result is included only once the
    job is terminal (completed/failed), which the client fetches a single time.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        if job.get("status") in ("completed", "failed"):
            return dict(job)
        return {k: v for k, v in job.items() if k != "result"}


def cleanup_old_jobs():
    """Delete jobs older than the TTL (called periodically and on startup)."""
    cutoff = time.time() - _TTL
    with _lock:
        to_delete = [k for k, v in _jobs.items() if v.get("created", 0) < cutoff]
        for k in to_delete:
            del _jobs[k]
        if to_delete:
            _flush()


# Purge anything already stale at import time so a long-idle restart starts clean.
cleanup_old_jobs()
