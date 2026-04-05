"""
app/storage/connector_store.py

Async PostgreSQL CRUD for Connector and ConnectorSyncRun.

All functions accept an AsyncSession and are flush-only (no commit).
The caller owns the transaction boundary.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sql.connector import Connector, ConnectorSyncRun


# ── Connector CRUD ────────────────────────────────────────────────────────────


async def create_connector(
    session: AsyncSession,
    *,
    name: str,
    connector_type: str,
    credentials_enc: str,
    config_json: str,
    sync_schedule: str = "0 * * * *",
    is_enabled: bool = True,
) -> Connector:
    """Insert a new Connector row and return the persisted instance."""
    connector = Connector(
        name=name,
        connector_type=connector_type,
        credentials_enc=credentials_enc,
        config=config_json,
        sync_schedule=sync_schedule,
        is_enabled=is_enabled,
    )
    session.add(connector)
    await session.flush()
    await session.refresh(connector)
    return connector


async def get_connector(
    session: AsyncSession,
    connector_id: uuid.UUID,
) -> Connector | None:
    """Fetch a single connector by primary key."""
    stmt = select(Connector).where(Connector.id == connector_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_connectors(
    session: AsyncSession,
    *,
    offset: int = 0,
    limit: int = 50,
    enabled_only: bool = False,
) -> tuple[list[Connector], int]:
    """Return a paginated (connectors, total_count) tuple.

    Results are ordered by created_at descending (newest first).
    """
    stmt = select(Connector)
    if enabled_only:
        stmt = stmt.where(Connector.is_enabled.is_(True))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await session.execute(count_stmt)).scalar_one()

    data_stmt = (
        stmt
        .order_by(Connector.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = (await session.execute(data_stmt)).scalars().all()
    return list(rows), total


async def update_connector(
    session: AsyncSession,
    connector_id: uuid.UUID,
    **fields: Any,
) -> Connector | None:
    """Apply arbitrary field updates to a connector.

    Only columns present in ``fields`` are changed.
    Returns the updated instance, or None if not found.
    """
    connector = await get_connector(session, connector_id)
    if connector is None:
        return None
    for key, value in fields.items():
        if hasattr(connector, key):
            setattr(connector, key, value)
    connector.updated_at = datetime.utcnow()
    await session.flush()
    await session.refresh(connector)
    return connector


async def delete_connector(
    session: AsyncSession,
    connector_id: uuid.UUID,
) -> bool:
    """Delete a connector by primary key.

    Returns True if a row was deleted, False if not found.
    CASCADE on connector_sync_runs handles cleanup automatically.
    """
    stmt = (
        delete(Connector)
        .where(Connector.id == connector_id)
        .returning(Connector.id)
    )
    result = await session.execute(stmt)
    await session.flush()
    return result.scalar_one_or_none() is not None


# ── ConnectorSyncRun CRUD ─────────────────────────────────────────────────────


async def create_sync_run(
    session: AsyncSession,
    connector_id: uuid.UUID,
) -> ConnectorSyncRun:
    """Insert a new sync run with status='running' and return it."""
    run = ConnectorSyncRun(
        connector_id=connector_id,
        status="running",
    )
    session.add(run)
    await session.flush()
    await session.refresh(run)
    return run


async def update_sync_run(
    session: AsyncSession,
    run_id: uuid.UUID,
    **fields: Any,
) -> ConnectorSyncRun | None:
    """Apply arbitrary field updates to a sync run.

    Returns the updated instance, or None if not found.
    """
    stmt = select(ConnectorSyncRun).where(ConnectorSyncRun.id == run_id)
    result = await session.execute(stmt)
    run = result.scalar_one_or_none()
    if run is None:
        return None
    for key, value in fields.items():
        if hasattr(run, key):
            setattr(run, key, value)
    await session.flush()
    await session.refresh(run)
    return run


async def list_sync_runs(
    session: AsyncSession,
    connector_id: uuid.UUID,
    *,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[ConnectorSyncRun], int]:
    """Return a paginated (runs, total_count) tuple for a given connector.

    Results are ordered by started_at descending (most recent first).
    """
    stmt = select(ConnectorSyncRun).where(
        ConnectorSyncRun.connector_id == connector_id
    )

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await session.execute(count_stmt)).scalar_one()

    data_stmt = (
        stmt
        .order_by(ConnectorSyncRun.started_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = (await session.execute(data_stmt)).scalars().all()
    return list(rows), total


async def get_latest_run_for_connector(
    session: AsyncSession,
    connector_id: uuid.UUID,
) -> ConnectorSyncRun | None:
    """Return the most recent sync run for a connector, or None."""
    stmt = (
        select(ConnectorSyncRun)
        .where(ConnectorSyncRun.connector_id == connector_id)
        .order_by(ConnectorSyncRun.started_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_latest_successful_run(
    session: AsyncSession,
    connector_id: uuid.UUID,
) -> ConnectorSyncRun | None:
    """Return the most recent successful sync run, used to read the cursor."""
    stmt = (
        select(ConnectorSyncRun)
        .where(
            ConnectorSyncRun.connector_id == connector_id,
            ConnectorSyncRun.status == "success",
        )
        .order_by(ConnectorSyncRun.started_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
