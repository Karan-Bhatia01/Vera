"""
Job store for tracking async ML pipeline training.

Provides in-memory job state tracking with threading-safe access.
Jobs persist until 1 hour after completion.
"""

import threading
import uuid
import time
from typing import Any

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


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
    return job_id


def update_job(job_id: str, **kwargs):
    """Update job state."""
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> dict | None:
    """Get job state (returns a copy)."""
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def cleanup_old_jobs():
    """Delete jobs older than 1 hour (called periodically)."""
    cutoff = time.time() - 3600
    with _lock:
        to_delete = [k for k, v in _jobs.items() if v["created"] < cutoff]
        for k in to_delete:
            del _jobs[k]
