"""Celery tasks for connector syncing and schedule polling."""

from __future__ import annotations

import asyncio
import uuid

import structlog

from app.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)


# ── Sync task ─────────────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="connector.sync",
    max_retries=3,
    default_retry_delay=60,
)
def sync_connector_task(self, connector_id: str) -> dict:
    """Run a full sync cycle for a single connector.

    Wraps the async ``run_sync()`` orchestrator for Celery's sync task runner.
    Retries up to 3 times on unexpected failures with a 60-second delay.
    """
    try:
        return asyncio.run(_async_run_sync(connector_id))
    except Exception as exc:
        logger.error("sync_connector_task_failed", connector_id=connector_id, error=str(exc))
        raise self.retry(exc=exc)


async def _async_run_sync(connector_id: str) -> dict:
    """Initialize services, run sync, and clean up."""
    from app.ingestion.producer import init_kafka_producer, close_kafka_producer
    from app.ingestion.sync_service import run_sync

    await init_kafka_producer()
    try:
        result = await run_sync(uuid.UUID(connector_id))
        return result.as_dict()
    finally:
        await close_kafka_producer()


# ── Poll-and-schedule task ────────────────────────────────────────────────────


@celery_app.task(name="connector.poll_and_schedule")
def poll_and_schedule_connectors() -> None:
    """Check all enabled connectors and enqueue syncs for those that are due.

    Runs every 60 seconds via Celery Beat. Uses croniter to evaluate each
    connector's cron schedule against its last_synced_at timestamp.
    """
    asyncio.run(_async_poll_and_schedule())


async def _async_poll_and_schedule() -> None:
    from datetime import datetime, timezone

    from croniter import croniter

    from app.database.postgres import async_session_factory
    from app.storage import connector_store

    now = datetime.now(timezone.utc)

    async with async_session_factory() as session:
        connectors, _ = await connector_store.list_connectors(
            session, enabled_only=True, limit=500
        )

        for connector in connectors:
            # Skip if a sync is already in progress
            latest_run = await connector_store.get_latest_run_for_connector(
                session, connector.id
            )
            if latest_run and latest_run.status == "running":
                logger.debug(
                    "connector_sync_already_running", connector_id=str(connector.id)
                )
                continue

            # Never synced → enqueue immediately
            if connector.last_synced_at is None:
                _enqueue(connector.id)
                continue

            # Check cron schedule
            try:
                cron = croniter(connector.sync_schedule, connector.last_synced_at)
                next_run: datetime = cron.get_next(datetime)
                if next_run <= now:
                    _enqueue(connector.id)
            except Exception as exc:
                logger.warning(
                    "connector_schedule_parse_failed",
                    connector_id=str(connector.id),
                    schedule=connector.sync_schedule,
                    error=str(exc),
                )


def _enqueue(connector_id: uuid.UUID) -> None:
    sync_connector_task.delay(str(connector_id))
    logger.info("connector_sync_enqueued", connector_id=str(connector_id))
