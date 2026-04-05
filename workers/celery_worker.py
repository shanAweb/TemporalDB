"""Celery worker entry point.

Run the worker:
    celery -A workers.celery_worker.celery_app worker --loglevel=info

Run the beat scheduler (separate process):
    celery -A workers.celery_worker.celery_app beat --loglevel=info

Or run both together (development only):
    celery -A workers.celery_worker.celery_app worker --beat --loglevel=info
"""

from app.tasks.celery_app import celery_app
from app.tasks import connector_tasks  # noqa: F401 — registers tasks with the app

__all__ = ["celery_app"]
