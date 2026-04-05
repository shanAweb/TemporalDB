"""Celery application instance and configuration."""

from celery import Celery

from app.config import settings

celery_app = Celery("temporaldb")

celery_app.config_from_object(
    {
        "broker_url": settings.celery_broker_url,
        "result_backend": settings.celery_result_backend,
        "task_serializer": "json",
        "result_serializer": "json",
        "accept_content": ["json"],
        "timezone": "UTC",
        "enable_utc": True,
        # Beat schedule — one entry polls all enabled connectors every 60 seconds
        # and enqueues individual sync tasks for those that are due.
        "beat_schedule": {
            "poll-connector-schedules": {
                "task": "connector.poll_and_schedule",
                "schedule": 60.0,
            },
        },
        "beat_scheduler": "celery.beat:PersistentScheduler",
        "beat_schedule_filename": "/tmp/celerybeat-schedule",
        # Worker settings
        "worker_prefetch_multiplier": 1,
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
    }
)
