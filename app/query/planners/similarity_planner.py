"""
app/query/planners/similarity_planner.py

Planner for SIMILARITY queries.

Strategy
--------
1. Embed the user's question with the sentence-transformer model.
2. Run pgvector cosine similarity search against the events table.
3. Optionally post-filter by entity_id (Python-side, since pgvector does
   not support joins in the ANN scan) and time_range.

The distance score is inverted to a [0, 1] confidence where distance 0
(identical) → confidence 1.0 and distance 1.0 (orthogonal) → confidence 0.0.
"""
from __future__ import annotations

import uuid
from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas.query import TimeRange
from app.models.sql.event import Event
from app.nlp.embedder import embed
from app.query.planners import PlanResult
from app.storage import event_store

logger = structlog.get_logger(__name__)

_DEFAULT_LIMIT = 10
_MAX_DISTANCE  = 0.9   # cosine distance ceiling — filters very distant results


async def run(
    pg_session: AsyncSession,
    question: str,
    *,
    entity_id: uuid.UUID | None = None,
    time_range: TimeRange | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> PlanResult:
    """Execute the similarity planning strategy for *question*.

    Args:
        pg_session:  SQLAlchemy async session (read-only).
        question:    Raw natural-language question to embed and search.
        entity_id:   Optional entity UUID; results not involving this entity
                     are discarded after the vector search.
        time_range:  Optional UTC time window; events outside the window are
                     discarded after the vector search.
        limit:       Maximum number of events to return.

    Returns:
        PlanResult populated with semantically similar events (no causal_chain).
    """
    embedding = await embed(question)

    # Fetch more than *limit* to allow for post-filtering headroom.
    fetch_limit = limit * 3 if (entity_id or time_range) else limit

    pairs: list[tuple[Event, float]] = await event_store.similarity_search(
        pg_session,
        embedding,
        limit=fetch_limit,
        max_distance=_MAX_DISTANCE,
    )

    # ── Post-filter ────────────────────────────────────────────────────────
    if entity_id is not None:
        from sqlalchemy import select
        from app.models.sql.event_entity import EventEntity

        linked_stmt = select(EventEntity.event_id).where(
            EventEntity.entity_id == entity_id
        )
        linked_ids: set[uuid.UUID] = set(
            (await pg_session.execute(linked_stmt)).scalars().all()
        )
        pairs = [(ev, dist) for ev, dist in pairs if ev.id in linked_ids]

    if time_range is not None:
        def _in_range(ev: Event) -> bool:
            if ev.ts_start is None:
                return False
            ts: datetime = ev.ts_start
            return time_range.start <= ts <= time_range.end

        pairs = [(ev, dist) for ev, dist in pairs if _in_range(ev)]

    # Trim to requested limit after filtering.
    pairs = pairs[:limit]

    events = [ev for ev, _dist in pairs]
    distances = [dist for _ev, dist in pairs]

    document_ids = {ev.document_id for ev in events}

    # Aggregate confidence: average of (1 - distance) across results.
    if distances:
        confidence = round(sum(1.0 - d for d in distances) / len(distances), 4)
    else:
        confidence = 0.0

    logger.info(
        "similarity_planner_complete",
        returned=len(events),
        avg_confidence=confidence,
    )
    return PlanResult(
        events=events,
        causal_chain=[],
        document_ids=document_ids,
        confidence=confidence,
    )
