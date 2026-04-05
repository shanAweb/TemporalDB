"""
app/api/routes/events.py

Event browsing endpoints.

GET /events
    Paginated list of events with optional filters for entity, document,
    time range, and event type.

GET /events/{event_id}
    Single event by UUID, with eager-loaded entity associations.
"""
from __future__ import annotations

import uuid
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import require_api_key
from app.database.postgres import get_db
from app.models.schemas.event import EventListResponse, EventOut
from app.storage import event_store

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get(
    "",
    response_model=EventListResponse,
    summary="List events",
)
async def list_events(
    entity_id: uuid.UUID | None = Query(default=None, description="Filter by entity UUID"),
    document_id: uuid.UUID | None = Query(default=None, description="Filter by document UUID"),
    from_date: datetime | None = Query(default=None, description="Events on or after this UTC datetime"),
    to_date: datetime | None = Query(default=None, description="Events on or before this UTC datetime"),
    event_type: str | None = Query(default=None, description="Filter by event type (action, state_change, …)"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results to return"),
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> EventListResponse:
    """Return a paginated list of events matching the supplied filters.

    All filters are optional and combined with AND logic.  Results are
    ordered by ``ts_start`` ascending (NULL timestamps last), then by
    ``created_at``.
    """
    events, total = await event_store.list_events(
        pg_session,
        entity_id=entity_id,
        document_id=document_id,
        from_date=from_date,
        to_date=to_date,
        event_type=event_type,
        offset=offset,
        limit=limit,
    )

    logger.info(
        "events_listed",
        total=total,
        returned=len(events),
        offset=offset,
        limit=limit,
    )
    return EventListResponse(
        events=[EventOut.model_validate(ev) for ev in events],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/{event_id}",
    response_model=EventOut,
    summary="Get event by ID",
)
async def get_event(
    event_id: uuid.UUID,
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> EventOut:
    """Fetch a single event by its UUID.

    Returns HTTP 404 when no event with the given ID exists.
    """
    event = await event_store.get_event_by_id(pg_session, event_id)
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found.",
        )
    logger.info("event_fetched", event_id=str(event_id))
    return EventOut.model_validate(event)
