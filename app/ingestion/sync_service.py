"""
app/ingestion/sync_service.py

Connector sync orchestrator.

``run_sync(connector_id)`` is the single entry point called by both
the Celery task (Phase 4) and the "Sync Now" API endpoint (Phase 6).

Flow
----
1. Load Connector row from DB; decrypt credentials.
2. Read incremental cursor from the last successful run.
3. Create a ConnectorSyncRun with status="running".
4. Iterate fetch_items() → transform → normalize → deduplicate → persist → Kafka.
5. Finalize the run row and update connector.last_synced_at / last_sync_status.

Deduplication key
-----------------
Each item's source is set to ``"{connector_type}:{external_id}:{item_type}"``
(e.g. ``"jira:ENG-42:issue"``).  If the item content changes between syncs
the SHA-256 fingerprint changes, producing a new Document while the previous
version is preserved for temporal analysis.
"""
from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog
from redis.asyncio import Redis

from app.config import settings
from app.database.postgres import async_session_factory
from app.ingestion.connectors.registry import get_connector
from app.ingestion.deduplicator import check_and_register, compute_fingerprint
from app.ingestion.normalizer import normalize
from app.ingestion.producer import publish_document_ingested
from app.models.sql.document import Document
from app.storage import connector_store
from app.utils.crypto import decrypt_credentials

logger = structlog.get_logger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class ConnectorSyncResult:
    """Summary returned by run_sync()."""

    connector_id: uuid.UUID
    run_id: uuid.UUID
    status: str
    items_fetched: int
    items_ingested: int
    items_skipped: int
    error_message: str | None = None

    def as_dict(self) -> dict:
        return {
            "connector_id": str(self.connector_id),
            "run_id": str(self.run_id),
            "status": self.status,
            "items_fetched": self.items_fetched,
            "items_ingested": self.items_ingested,
            "items_skipped": self.items_skipped,
            "error_message": self.error_message,
        }


# ── Main entry point ──────────────────────────────────────────────────────────


async def run_sync(connector_id: uuid.UUID) -> ConnectorSyncResult:
    """Run a full sync cycle for the given connector.

    Safe to call concurrently for different connectors; the same connector
    should not be run in parallel (Celery Beat ensures this by design).
    """
    log = logger.bind(connector_id=str(connector_id))

    # ── 1. Load connector and decrypt credentials ─────────────────────────────
    async with async_session_factory() as session:
        connector = await connector_store.get_connector(session, connector_id)

    if connector is None:
        raise ValueError(f"Connector {connector_id} not found.")
    if not connector.is_enabled:
        raise ValueError(f"Connector {connector_id} is disabled.")

    try:
        credentials = decrypt_credentials(connector.credentials_enc or "")
    except ValueError as exc:
        raise RuntimeError(f"Failed to decrypt credentials: {exc}") from exc

    config: dict = json.loads(connector.config or "{}")

    # ── 2. Read incremental cursor ────────────────────────────────────────────
    async with async_session_factory() as session:
        last_ok_run = await connector_store.get_latest_successful_run(
            session, connector_id
        )

    cursor: str | None = None
    if last_ok_run and last_ok_run.metadata_:
        try:
            cursor = json.loads(last_ok_run.metadata_).get("cursor")
        except (json.JSONDecodeError, AttributeError):
            cursor = None

    log.info("connector_sync_starting", connector_type=connector.connector_type, cursor=cursor)

    # ── 3. Create sync run ────────────────────────────────────────────────────
    async with async_session_factory() as session:
        sync_run = await connector_store.create_sync_run(session, connector_id)
        await session.commit()
        run_id = sync_run.id

    # ── 4. Fetch → transform → normalize → deduplicate → persist ─────────────
    connector_instance = get_connector(connector.connector_type)
    # Create a self-contained Redis client — no pool init required in worker context
    redis = Redis.from_url(settings.redis_url, decode_responses=True)

    items_fetched = 0
    items_ingested = 0
    items_skipped = 0
    error_message: str | None = None
    status = "success"
    new_cursor: str | None = None

    try:
        async for raw_item in connector_instance.fetch_items(credentials, config, cursor):
            items_fetched += 1

            # Update rolling cursor for incremental next run
            new_cursor = _build_cursor(connector.connector_type, raw_item, new_cursor)

            # Transform
            try:
                result = connector_instance.transform_item(raw_item, connector_id)
            except Exception as exc:
                log.warning(
                    "connector_transform_failed",
                    external_id=raw_item.external_id,
                    item_type=raw_item.item_type,
                    error=str(exc),
                )
                items_skipped += 1
                continue

            # Normalize
            normalized_text = normalize(result.text)
            if not normalized_text.strip():
                items_skipped += 1
                continue

            # Deduplicate
            doc_id_str = str(uuid.uuid4())
            dedup = await check_and_register(normalized_text, doc_id_str, redis)
            if dedup.is_duplicate:
                items_skipped += 1
                continue

            # Persist Document
            source: str = result.metadata.get(
                "source",
                f"{connector.connector_type}:{raw_item.external_id}:{raw_item.item_type}",
            )
            fingerprint = compute_fingerprint(normalized_text)

            doc = Document(
                id=uuid.UUID(doc_id_str),
                source=source,
                filename=result.filename,
                content_hash=fingerprint,
                raw_text=normalized_text,
                mime_type="text/plain",
                metadata_=json.dumps(result.metadata),
            )
            async with async_session_factory() as doc_session:
                doc_session.add(doc)
                await doc_session.commit()

            # Publish to Kafka → NLP worker picks up automatically
            await publish_document_ingested(
                document_id=doc_id_str,
                source=source,
                filename=result.filename,
                text=normalized_text,
            )

            items_ingested += 1
            log.debug(
                "connector_item_ingested",
                external_id=raw_item.external_id,
                item_type=raw_item.item_type,
                document_id=doc_id_str,
            )

    except Exception as exc:
        status = "error"
        error_message = traceback.format_exc()
        log.error(
            "connector_sync_failed",
            connector_type=connector.connector_type,
            error=str(exc),
        )

    finally:
        await redis.aclose()

    # ── 5. Finalize run and update connector ──────────────────────────────────
    finished_at = datetime.now(timezone.utc)
    run_metadata = json.dumps({"cursor": new_cursor}) if new_cursor else None

    async with async_session_factory() as session:
        await connector_store.update_sync_run(
            session,
            run_id,
            finished_at=finished_at,
            status=status,
            items_fetched=items_fetched,
            items_ingested=items_ingested,
            items_skipped=items_skipped,
            error_message=error_message,
            metadata_=run_metadata,
        )
        await connector_store.update_connector(
            session,
            connector_id,
            last_synced_at=finished_at,
            last_sync_status=status,
        )
        await session.commit()

    log.info(
        "connector_sync_complete",
        connector_type=connector.connector_type,
        status=status,
        items_fetched=items_fetched,
        items_ingested=items_ingested,
        items_skipped=items_skipped,
    )

    return ConnectorSyncResult(
        connector_id=connector_id,
        run_id=run_id,
        status=status,
        items_fetched=items_fetched,
        items_ingested=items_ingested,
        items_skipped=items_skipped,
        error_message=error_message,
    )


# ── Cursor helpers ────────────────────────────────────────────────────────────


def _build_cursor(
    connector_type: str,
    raw_item: object,
    current_cursor: str | None,
) -> str | None:
    """Return an updated cursor string after processing a raw item.

    Jira   — ISO timestamp of the issue's 'updated' field (keep the latest seen).
    ClickUp — Unix ms timestamp string of the task's 'date_updated' field.
    Time Doctor — ISO date string of the worklog's 'date' field (keep the latest).
    All others — unchanged.
    """
    data: dict = getattr(raw_item, "data", {})
    item_type: str = getattr(raw_item, "item_type", "")

    if connector_type == "jira" and item_type == "issue":
        updated = data.get("fields", {}).get("updated", "")
        if updated:
            # Keep lexicographically largest ISO timestamp (latest)
            if current_cursor is None or updated > current_cursor:
                return updated

    elif connector_type == "clickup" and item_type == "task":
        date_updated = data.get("date_updated")
        if date_updated:
            if current_cursor is None or int(date_updated) > int(current_cursor):
                return str(date_updated)

    elif connector_type == "timedoctor" and item_type == "worklog":
        date = data.get("date", "")
        if date:
            if current_cursor is None or date > current_cursor:
                return date

    return current_cursor
