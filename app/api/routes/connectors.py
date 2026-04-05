"""
app/api/routes/connectors.py

Connector management endpoints.

GET    /connectors                      List all connectors (paginated)
POST   /connectors                      Create a connector
GET    /connectors/{id}                 Get one connector
PATCH  /connectors/{id}                 Update a connector
DELETE /connectors/{id}                 Delete a connector
POST   /connectors/{id}/sync            Trigger an immediate sync (202)
GET    /connectors/{id}/runs            List sync history (paginated)
GET    /connectors/{id}/runs/{run_id}   Get a single sync run
POST   /connectors/{id}/validate        Test credentials without persisting

All endpoints require X-API-Key authentication.
Credentials are encrypted at rest using AES-256-GCM; they are never
returned in any response.
"""
from __future__ import annotations

import json
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import require_api_key
from app.database.postgres import get_db
from app.ingestion.connectors.registry import get_connector
from app.models.schemas.connector import (
    ConnectorCreate,
    ConnectorListResponse,
    ConnectorOut,
    ConnectorUpdate,
    SyncRunListResponse,
    SyncRunOut,
    SyncTriggerResponse,
    ValidateCredentialsResponse,
)
from app.storage import connector_store
from app.utils.crypto import decrypt_credentials, encrypt_credentials

logger = structlog.get_logger(__name__)

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_connector_or_404(
    connector_id: uuid.UUID,
    session: AsyncSession,
) -> object:
    connector = await connector_store.get_connector(session, connector_id)
    if connector is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector {connector_id} not found.",
        )
    return connector


async def _build_connector_out(connector: object, session: AsyncSession) -> ConnectorOut:
    """Attach the latest sync run and return a ConnectorOut schema."""
    last_run = await connector_store.get_latest_run_for_connector(
        session, connector.id  # type: ignore[attr-defined]
    )
    out = ConnectorOut.model_validate(connector)
    out.last_run = SyncRunOut.model_validate(last_run) if last_run else None
    return out


# ── List connectors ───────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=ConnectorListResponse,
    summary="List connectors",
)
async def list_connectors(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> ConnectorListResponse:
    """Return a paginated list of all configured connectors."""
    connectors, total = await connector_store.list_connectors(
        pg_session, offset=offset, limit=limit
    )
    items = [await _build_connector_out(c, pg_session) for c in connectors]
    return ConnectorListResponse(connectors=items, total=total, offset=offset, limit=limit)


# ── Create connector ──────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=ConnectorOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a connector",
)
async def create_connector(
    body: ConnectorCreate,
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> ConnectorOut:
    """Create a new external connector.

    ``credentials`` are encrypted server-side before storage and are never
    returned in any response.
    """
    # Validate connector type is supported
    try:
        get_connector(body.connector_type)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        credentials_enc = encrypt_credentials(body.credentials)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Credential encryption unavailable: {exc}",
        )

    connector = await connector_store.create_connector(
        pg_session,
        name=body.name,
        connector_type=body.connector_type,
        credentials_enc=credentials_enc,
        config_json=json.dumps(body.config),
        sync_schedule=body.sync_schedule,
    )
    logger.info("connector_created", connector_id=str(connector.id), type=body.connector_type)
    return await _build_connector_out(connector, pg_session)


# ── Get connector ─────────────────────────────────────────────────────────────


@router.get(
    "/{connector_id}",
    response_model=ConnectorOut,
    summary="Get connector by ID",
)
async def get_connector_by_id(
    connector_id: uuid.UUID,
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> ConnectorOut:
    """Fetch a single connector by UUID. Credentials are never returned."""
    connector = await _get_connector_or_404(connector_id, pg_session)
    return await _build_connector_out(connector, pg_session)


# ── Update connector ──────────────────────────────────────────────────────────


@router.patch(
    "/{connector_id}",
    response_model=ConnectorOut,
    summary="Update a connector",
)
async def update_connector(
    connector_id: uuid.UUID,
    body: ConnectorUpdate,
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> ConnectorOut:
    """Partially update a connector. Only supplied fields are changed.

    If ``credentials`` is present in the body they are re-encrypted and
    replace the stored value.
    """
    await _get_connector_or_404(connector_id, pg_session)

    updates: dict = body.model_dump(exclude_none=True)

    # Re-encrypt credentials if provided
    if "credentials" in updates:
        try:
            updates["credentials_enc"] = encrypt_credentials(updates.pop("credentials"))
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Credential encryption unavailable: {exc}",
            )

    # Serialize config dict to JSON string for storage
    if "config" in updates:
        updates["config"] = json.dumps(updates["config"])

    connector = await connector_store.update_connector(pg_session, connector_id, **updates)
    logger.info("connector_updated", connector_id=str(connector_id))
    return await _build_connector_out(connector, pg_session)


# ── Delete connector ──────────────────────────────────────────────────────────


@router.delete(
    "/{connector_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a connector",
)
async def delete_connector(
    connector_id: uuid.UUID,
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> dict:
    """Delete a connector and all its sync history.

    This is irreversible. The CASCADE constraint on ``connector_sync_runs``
    removes all associated run records automatically.
    """
    deleted = await connector_store.delete_connector(pg_session, connector_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Connector {connector_id} not found.",
        )
    logger.info("connector_deleted", connector_id=str(connector_id))
    return {"deleted": True}


# ── Trigger sync ──────────────────────────────────────────────────────────────


@router.post(
    "/{connector_id}/sync",
    response_model=SyncTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger an immediate sync",
)
async def trigger_sync(
    connector_id: uuid.UUID,
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> SyncTriggerResponse:
    """Enqueue an immediate sync for the connector and return the run ID.

    Returns ``202 Accepted`` immediately — the sync runs asynchronously in
    the Celery worker. Poll ``GET /connectors/{id}/runs/{run_id}`` to track
    progress.
    """
    await _get_connector_or_404(connector_id, pg_session)

    # Create the run row now so the caller has a run_id to track immediately
    sync_run = await connector_store.create_sync_run(pg_session, connector_id)
    await pg_session.flush()

    # Enqueue Celery task, passing the pre-created run_id
    from app.tasks.connector_tasks import sync_connector_task
    sync_connector_task.delay(str(connector_id), str(sync_run.id))

    logger.info(
        "connector_sync_triggered",
        connector_id=str(connector_id),
        run_id=str(sync_run.id),
    )
    return SyncTriggerResponse(
        connector_id=connector_id,
        run_id=sync_run.id,
        message="Sync enqueued. Track progress via the run ID.",
    )


# ── Sync run history ──────────────────────────────────────────────────────────


@router.get(
    "/{connector_id}/runs",
    response_model=SyncRunListResponse,
    summary="List sync run history",
)
async def list_sync_runs(
    connector_id: uuid.UUID,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> SyncRunListResponse:
    """Return a paginated sync run history for a connector (newest first)."""
    await _get_connector_or_404(connector_id, pg_session)
    runs, total = await connector_store.list_sync_runs(
        pg_session, connector_id, offset=offset, limit=limit
    )
    return SyncRunListResponse(
        runs=[SyncRunOut.model_validate(r) for r in runs],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/{connector_id}/runs/{run_id}",
    response_model=SyncRunOut,
    summary="Get a single sync run",
)
async def get_sync_run(
    connector_id: uuid.UUID,
    run_id: uuid.UUID,
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> SyncRunOut:
    """Fetch detail for a single sync run, including item counts and any error."""
    await _get_connector_or_404(connector_id, pg_session)

    from sqlalchemy import select
    from app.models.sql.connector import ConnectorSyncRun

    stmt = select(ConnectorSyncRun).where(
        ConnectorSyncRun.id == run_id,
        ConnectorSyncRun.connector_id == connector_id,
    )
    result = await pg_session.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sync run {run_id} not found for connector {connector_id}.",
        )
    return SyncRunOut.model_validate(run)


# ── Validate credentials ──────────────────────────────────────────────────────


@router.post(
    "/{connector_id}/validate",
    response_model=ValidateCredentialsResponse,
    summary="Validate connector credentials",
)
async def validate_credentials(
    connector_id: uuid.UUID,
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> ValidateCredentialsResponse:
    """Test whether the stored credentials can reach the external API.

    No data is fetched or stored. Returns ``{"valid": true}`` on success or
    ``{"valid": false, "error": "..."}`` on failure.
    """
    connector = await _get_connector_or_404(connector_id, pg_session)

    try:
        credentials = decrypt_credentials(connector.credentials_enc or "")  # type: ignore[attr-defined]
    except ValueError as exc:
        return ValidateCredentialsResponse(valid=False, error=f"Cannot decrypt credentials: {exc}")

    try:
        connector_instance = get_connector(connector.connector_type)  # type: ignore[attr-defined]
    except ValueError as exc:
        return ValidateCredentialsResponse(valid=False, error=str(exc))

    valid, error = await connector_instance.validate_credentials(credentials)
    logger.info(
        "connector_credentials_validated",
        connector_id=str(connector_id),
        valid=valid,
    )
    return ValidateCredentialsResponse(valid=valid, error=error)
