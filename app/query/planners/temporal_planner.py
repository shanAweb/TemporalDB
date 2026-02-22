"""
app/query/planners/temporal_planner.py

Planner for TEMPORAL_RANGE queries.

Strategy
--------
Delegates entirely to event_store.list_events() with the resolved time
range applied as from_date / to_date filters.  An optional entity_id
further narrows results to events involving a specific entity.

Results are returned ordered by ts_start ascending (NULLs last) as
enforced by the store layer.

The planner is intentionally simple — all the complexity sits in the
temporal_extractor and entity_resolver that run before it.
"""
from __future__ import annotations

import uuid

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas.query import TimeRange
from app.query.planners import PlanResult
from app.storage import event_store

logger = structlog.get_logger(__name__)

# Maximum events returned for a temporal range query.
_DEFAULT_LIMIT = 50


async def run(
    pg_session: AsyncSession,
    time_range: TimeRange | None,
    *,
    entity_id: uuid.UUID | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> PlanResult:
    """Execute the temporal range planning strategy.

    Args:
        pg_session:  SQLAlchemy async session (read-only).
        time_range:  UTC time window to filter events (required for useful
                     results; if None the query returns all events up to
                     *limit*).
        entity_id:   Optional entity UUID to narrow results further.
        limit:       Maximum number of events to return.

    Returns:
        PlanResult populated with matching events (no causal_chain).
    """
    from_date = time_range.start if time_range else None
    to_date   = time_range.end   if time_range else None

    events, total = await event_store.list_events(
        pg_session,
        from_date=from_date,
        to_date=to_date,
        entity_id=entity_id,
        limit=limit,
        offset=0,
    )

    document_ids = {ev.document_id for ev in events}

    # Confidence scales with how specific the time window is.
    # A wide-open range (no time_range) is less confident.
    confidence = 0.85 if time_range else 0.60

    logger.info(
        "temporal_planner_complete",
        from_date=str(from_date),
        to_date=str(to_date),
        returned=len(events),
        total=total,
    )
    return PlanResult(
        events=events,
        causal_chain=[],
        document_ids=document_ids,
        confidence=confidence,
    )
