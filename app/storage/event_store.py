"""
app/storage/event_store.py

Async PostgreSQL CRUD for Events and CausalRelations.

All functions accept an AsyncSession and are flush-only (no commit).
The caller (API route or Celery task) owns the transaction boundary.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.sql.causal_relation import CausalRelation
from app.models.sql.event import Event
from app.models.sql.event_entity import EventEntity


# ── Event CRUD ────────────────────────────────────────────────────────────────


async def insert_event(session: AsyncSession, **kwargs: Any) -> Event:
    """
    Insert a new Event row and return the persisted instance.

    Expected kwargs mirror Event columns:
        description, event_type, ts_start, ts_end, confidence,
        source_sentence, embedding, document_id.
    """
    event = Event(**kwargs)
    session.add(event)
    await session.flush()
    await session.refresh(event)
    return event


async def get_event_by_id(
    session: AsyncSession,
    event_id: uuid.UUID,
    *,
    load_entities: bool = False,
) -> Event | None:
    """
    Fetch a single event by primary key.

    Set load_entities=True to eager-load the related Entity objects
    via the event_entities join table.
    """
    stmt = select(Event).where(Event.id == event_id)
    if load_entities:
        stmt = stmt.options(selectinload(Event.entities))
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_events(
    session: AsyncSession,
    *,
    document_id: uuid.UUID | None = None,
    entity_id: uuid.UUID | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    event_type: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[Event], int]:
    """
    Return a paginated (results, total_count) tuple matching the given filters.

    All filters are optional and combined with AND logic.
    Results are ordered by ts_start ascending (NULLs last), then created_at.
    """
    stmt = select(Event)

    if entity_id is not None:
        stmt = stmt.join(
            EventEntity, EventEntity.event_id == Event.id
        ).where(EventEntity.entity_id == entity_id)

    if document_id is not None:
        stmt = stmt.where(Event.document_id == document_id)
    if from_date is not None:
        stmt = stmt.where(Event.ts_start >= from_date)
    if to_date is not None:
        stmt = stmt.where(Event.ts_start <= to_date)
    if event_type is not None:
        stmt = stmt.where(Event.event_type == event_type)

    # Total count using the same filters
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await session.execute(count_stmt)).scalar_one()

    # Paginated fetch
    data_stmt = (
        stmt
        .order_by(Event.ts_start.asc().nulls_last(), Event.created_at.asc())
        .offset(offset)
        .limit(limit)
    )
    rows = (await session.execute(data_stmt)).scalars().all()
    return list(rows), total


async def similarity_search(
    session: AsyncSession,
    embedding: list[float],
    *,
    limit: int = 10,
    max_distance: float = 1.0,
) -> list[tuple[Event, float]]:
    """
    Return events ranked by cosine distance to the query embedding.

    Cosine distance ranges from 0 (identical) to 2 (opposite).
    max_distance filters out events beyond that threshold (default 1.0
    keeps everything in the same hemisphere as the query vector).

    Returns a list of (Event, distance) tuples ordered nearest-first.
    """
    distance_expr = Event.embedding.cosine_distance(embedding)

    stmt = (
        select(Event, distance_expr.label("distance"))
        .where(Event.embedding.is_not(None))
        .where(distance_expr <= max_distance)
        .order_by(distance_expr.asc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [(row.Event, float(row.distance)) for row in rows]


async def delete_event(session: AsyncSession, event_id: uuid.UUID) -> bool:
    """
    Delete an event by primary key.

    Returns True if a row was deleted, False if the event did not exist.
    CASCADE constraints on event_entities and causal_relations handle cleanup.
    """
    stmt = delete(Event).where(Event.id == event_id).returning(Event.id)
    result = await session.execute(stmt)
    await session.flush()
    return result.scalar_one_or_none() is not None


# ── Entity linking ────────────────────────────────────────────────────────────


async def link_entities_to_event(
    session: AsyncSession,
    event_id: uuid.UUID,
    entity_ids: list[uuid.UUID],
) -> None:
    """
    Create event_entities rows linking the given entity UUIDs to an event.

    Skips IDs that are already linked (idempotent).
    """
    if not entity_ids:
        return

    # Fetch existing links to avoid duplicate-key errors
    existing_stmt = select(EventEntity.entity_id).where(
        EventEntity.event_id == event_id
    )
    existing: set[uuid.UUID] = set(
        (await session.execute(existing_stmt)).scalars().all()
    )

    new_links = [
        EventEntity(event_id=event_id, entity_id=eid)
        for eid in entity_ids
        if eid not in existing
    ]
    if new_links:
        session.add_all(new_links)
        await session.flush()


# ── Causal Relation CRUD ──────────────────────────────────────────────────────


async def insert_causal_relation(
    session: AsyncSession,
    *,
    cause_event_id: uuid.UUID,
    effect_event_id: uuid.UUID,
    confidence: float = 1.0,
    evidence: str | None = None,
) -> CausalRelation:
    """
    Insert a causal relation between two events.

    cause_event_id  → the event that caused something.
    effect_event_id → the event that was caused.
    confidence      → extraction confidence score (0.0–1.0).
    evidence        → the source sentence that supports this relation.
    """
    relation = CausalRelation(
        cause_event_id=cause_event_id,
        effect_event_id=effect_event_id,
        confidence=confidence,
        evidence=evidence,
    )
    session.add(relation)
    await session.flush()
    await session.refresh(relation)
    return relation


async def get_causal_relations(
    session: AsyncSession,
    event_id: uuid.UUID,
    *,
    as_cause: bool = True,
    as_effect: bool = True,
) -> list[CausalRelation]:
    """
    Fetch causal relations involving the given event.

    as_cause=True  → include relations where this event is the cause.
    as_effect=True → include relations where this event is the effect.
    Both flags can be True simultaneously (returns both directions).
    """
    if not as_cause and not as_effect:
        return []

    conditions = []
    if as_cause:
        conditions.append(CausalRelation.cause_event_id == event_id)
    if as_effect:
        conditions.append(CausalRelation.effect_event_id == event_id)

    stmt = select(CausalRelation).where(or_(*conditions))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def delete_causal_relation(
    session: AsyncSession,
    relation_id: uuid.UUID,
) -> bool:
    """
    Delete a causal relation by primary key.

    Returns True if deleted, False if not found.
    """
    stmt = (
        delete(CausalRelation)
        .where(CausalRelation.id == relation_id)
        .returning(CausalRelation.id)
    )
    result = await session.execute(stmt)
    await session.flush()
    return result.scalar_one_or_none() is not None
