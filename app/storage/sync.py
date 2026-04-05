"""
app/storage/sync.py

PostgreSQL → Neo4j synchronisation.

Two entry points
----------------
sync_document(pg_session, neo4j_session, document_id)
    Syncs every event, entity, and causal relation belonging to a single
    document.  Called immediately after the NLP pipeline finishes processing
    so the graph store reflects the new data without delay.

sync_all(pg_session, neo4j_session, *, batch_size)
    Full re-sync of all rows across events, entities, event_entities, and
    causal_relations.  Intended for recovery runs (e.g. after a Neo4j
    outage) or initial graph population from an existing Postgres dataset.

Both functions are idempotent — they call graph_store MERGE operations so
running them more than once is safe.

SyncResult
----------
Both functions return a SyncResult dataclass with counters for each object
type written to Neo4j.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog
from neo4j import AsyncSession as Neo4jSession
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.sql.causal_relation import CausalRelation
from app.models.sql.entity import Entity
from app.models.sql.event import Event
from app.models.sql.event_entity import EventEntity
from app.storage import graph_store

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Counts of objects written to Neo4j during a sync run."""

    event_nodes: int = 0
    entity_nodes: int = 0
    involves_edges: int = 0
    causal_edges: int = 0

    def __add__(self, other: SyncResult) -> SyncResult:
        return SyncResult(
            event_nodes    = self.event_nodes    + other.event_nodes,
            entity_nodes   = self.entity_nodes   + other.entity_nodes,
            involves_edges = self.involves_edges + other.involves_edges,
            causal_edges   = self.causal_edges   + other.causal_edges,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "event_nodes":    self.event_nodes,
            "entity_nodes":   self.entity_nodes,
            "involves_edges": self.involves_edges,
            "causal_edges":   self.causal_edges,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _sync_events(
    events: list[Event],
    neo4j_session: Neo4jSession,
) -> int:
    """Upsert a list of Event ORM instances as :Event nodes. Returns count."""
    for event in events:
        await graph_store.upsert_event_node(
            neo4j_session,
            event_id=event.id,
            description=event.description,
            event_type=event.event_type,
            ts_start=event.ts_start,
            ts_end=event.ts_end,
            confidence=event.confidence,
            source_sentence=event.source_sentence,
            document_id=event.document_id,
        )
    return len(events)


async def _sync_entities(
    entities: list[Entity],
    neo4j_session: Neo4jSession,
) -> int:
    """Upsert a list of Entity ORM instances as :Entity nodes. Returns count."""
    for entity in entities:
        await graph_store.upsert_entity_node(
            neo4j_session,
            entity_id=entity.id,
            name=entity.name,
            canonical_name=entity.canonical_name,
            entity_type=entity.type,
        )
    return len(entities)


async def _sync_involves(
    links: list[EventEntity],
    neo4j_session: Neo4jSession,
) -> int:
    """Upsert INVOLVES edges from EventEntity join rows. Returns count."""
    for link in links:
        await graph_store.upsert_involves_edge(
            neo4j_session,
            event_id=link.event_id,
            entity_id=link.entity_id,
        )
    return len(links)


async def _sync_causal(
    relations: list[CausalRelation],
    neo4j_session: Neo4jSession,
) -> int:
    """Upsert CAUSES edges from CausalRelation ORM instances. Returns count."""
    for rel in relations:
        await graph_store.upsert_causal_edge(
            neo4j_session,
            cause_event_id=rel.cause_event_id,
            effect_event_id=rel.effect_event_id,
            relation_id=rel.id,
            confidence=rel.confidence,
            evidence=rel.evidence,
        )
    return len(relations)


# ---------------------------------------------------------------------------
# Document-scoped sync  (called by the NLP worker after each document)
# ---------------------------------------------------------------------------

async def sync_document(
    pg_session: AsyncSession,
    neo4j_session: Neo4jSession,
    document_id: uuid.UUID,
) -> SyncResult:
    """Sync all graph data for a single document from Postgres to Neo4j.

    Loads events (with their entity and causal-relation sub-graphs) that
    belong to *document_id* and pushes every node and edge to Neo4j using
    MERGE semantics.

    Call order matters:
    1. Event nodes first (edges reference them by ID).
    2. Entity nodes (INVOLVES edges reference them).
    3. INVOLVES edges.
    4. CAUSES edges (both endpoint nodes must already exist).

    Args:
        pg_session:    SQLAlchemy async session (read-only within this call).
        neo4j_session: Neo4j async session.
        document_id:   UUID of the document whose data should be synced.

    Returns:
        SyncResult with per-type write counts.
    """
    result = SyncResult()

    # ── 1. Events (eager-load entities for INVOLVES edges) ──────────────────
    event_stmt = (
        select(Event)
        .where(Event.document_id == document_id)
        .options(selectinload(Event.entities))
    )
    events: list[Event] = list(
        (await pg_session.execute(event_stmt)).scalars().all()
    )

    if not events:
        logger.info("sync_document_no_events", document_id=str(document_id))
        return result

    event_ids = [e.id for e in events]

    result.event_nodes = await _sync_events(events, neo4j_session)

    # ── 2. Entities referenced by this document's events ───────────────────
    # Collect the unique entity set from the eagerly loaded relationship.
    seen_entity_ids: set[uuid.UUID] = set()
    unique_entities: list[Entity] = []
    for event in events:
        for entity in event.entities:
            if entity.id not in seen_entity_ids:
                seen_entity_ids.add(entity.id)
                unique_entities.append(entity)

    result.entity_nodes = await _sync_entities(unique_entities, neo4j_session)

    # ── 3. INVOLVES edges ───────────────────────────────────────────────────
    involves_stmt = select(EventEntity).where(
        EventEntity.event_id.in_(event_ids)
    )
    involves_rows: list[EventEntity] = list(
        (await pg_session.execute(involves_stmt)).scalars().all()
    )
    result.involves_edges = await _sync_involves(involves_rows, neo4j_session)

    # ── 4. CAUSES edges ─────────────────────────────────────────────────────
    causal_stmt = select(CausalRelation).where(
        CausalRelation.cause_event_id.in_(event_ids)
    )
    causal_rows: list[CausalRelation] = list(
        (await pg_session.execute(causal_stmt)).scalars().all()
    )
    result.causal_edges = await _sync_causal(causal_rows, neo4j_session)

    logger.info(
        "sync_document_complete",
        document_id=str(document_id),
        **result.as_dict(),
    )
    return result


# ---------------------------------------------------------------------------
# Full re-sync  (recovery / initial population)
# ---------------------------------------------------------------------------

async def sync_all(
    pg_session: AsyncSession,
    neo4j_session: Neo4jSession,
    *,
    batch_size: int = 500,
) -> SyncResult:
    """Re-sync every row in Postgres to Neo4j in batches.

    Processes Events, Entities, EventEntity links, and CausalRelations in
    separate paginated passes.  Each pass reads *batch_size* rows at a time
    to keep memory usage bounded for large datasets.

    This function is idempotent — safe to run multiple times or restart
    after interruption.

    Args:
        pg_session:    SQLAlchemy async session (read-only).
        neo4j_session: Neo4j async session.
        batch_size:    Rows per Postgres fetch (default 500).

    Returns:
        Aggregated SyncResult with total counts across all batches.
    """
    total = SyncResult()

    # ── Pass 1: Event nodes ─────────────────────────────────────────────────
    offset = 0
    while True:
        batch: list[Event] = list(
            (
                await pg_session.execute(
                    select(Event).order_by(Event.created_at.asc()).offset(offset).limit(batch_size)
                )
            ).scalars().all()
        )
        if not batch:
            break
        total.event_nodes += await _sync_events(batch, neo4j_session)
        logger.debug("sync_all_events_batch", offset=offset, count=len(batch))
        offset += len(batch)
        if len(batch) < batch_size:
            break

    # ── Pass 2: Entity nodes ────────────────────────────────────────────────
    offset = 0
    while True:
        batch_e: list[Entity] = list(
            (
                await pg_session.execute(
                    select(Entity).order_by(Entity.created_at.asc()).offset(offset).limit(batch_size)
                )
            ).scalars().all()
        )
        if not batch_e:
            break
        total.entity_nodes += await _sync_entities(batch_e, neo4j_session)
        logger.debug("sync_all_entities_batch", offset=offset, count=len(batch_e))
        offset += len(batch_e)
        if len(batch_e) < batch_size:
            break

    # ── Pass 3: INVOLVES edges ──────────────────────────────────────────────
    offset = 0
    while True:
        batch_i: list[EventEntity] = list(
            (
                await pg_session.execute(
                    select(EventEntity)
                    .order_by(EventEntity.event_id.asc(), EventEntity.entity_id.asc())
                    .offset(offset)
                    .limit(batch_size)
                )
            ).scalars().all()
        )
        if not batch_i:
            break
        total.involves_edges += await _sync_involves(batch_i, neo4j_session)
        logger.debug("sync_all_involves_batch", offset=offset, count=len(batch_i))
        offset += len(batch_i)
        if len(batch_i) < batch_size:
            break

    # ── Pass 4: CAUSES edges ────────────────────────────────────────────────
    offset = 0
    while True:
        batch_c: list[CausalRelation] = list(
            (
                await pg_session.execute(
                    select(CausalRelation).order_by(CausalRelation.created_at.asc()).offset(offset).limit(batch_size)
                )
            ).scalars().all()
        )
        if not batch_c:
            break
        total.causal_edges += await _sync_causal(batch_c, neo4j_session)
        logger.debug("sync_all_causal_batch", offset=offset, count=len(batch_c))
        offset += len(batch_c)
        if len(batch_c) < batch_size:
            break

    logger.info("sync_all_complete", **total.as_dict())
    return total
