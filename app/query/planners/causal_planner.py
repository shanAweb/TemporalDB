"""
app/query/planners/causal_planner.py

Planner for CAUSAL_WHY queries.

Strategy
--------
1. Embed the question and run pgvector similarity search to find seed events
   that are semantically closest to the query (the "anchor" events).

2. For each seed event, traverse the Neo4j causal graph in both directions
   up to *max_hops* hops using graph_store.get_causal_chain().

3. Collect the unique set of event IDs referenced in any chain, fetch their
   full details from PostgreSQL, and merge the traversal records into a
   single deduplicated causal chain ordered by hop distance.

4. If an entity_id filter is supplied, restrict the seed search to events
   that involve that entity (via the event_entities join table).

The planner does not synthesise an answer — it only gathers structured data
for the synthesizer.
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog
from neo4j import AsyncSession as Neo4jSession
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sql.event import Event
from app.models.sql.event_entity import EventEntity
from app.nlp.embedder import embed
from app.query.planners import PlanResult
from app.storage import event_store, graph_store

logger = structlog.get_logger(__name__)

# Number of seed events to use as causal graph entry points.
_SEED_LIMIT = 3
# Maximum cosine distance for seed similarity search.
_SEED_MAX_DISTANCE = 0.8


async def run(
    pg_session: AsyncSession,
    neo4j_session: Neo4jSession,
    question: str,
    *,
    entity_id: uuid.UUID | None = None,
    max_hops: int = 3,
) -> PlanResult:
    """Execute the causal planning strategy for *question*.

    Args:
        pg_session:    SQLAlchemy async session (read-only).
        neo4j_session: Neo4j async session.
        question:      Raw natural-language question from the user.
        entity_id:     Optional entity UUID to anchor the seed search.
        max_hops:      Maximum causal traversal depth (1 – 10).

    Returns:
        PlanResult populated with events and causal_chain.
    """
    # ── Step 1: find seed events via similarity search ─────────────────────
    question_embedding = await embed(question)

    if entity_id is not None:
        # Restrict similarity search to events involving the target entity.
        entity_event_ids_stmt = select(EventEntity.event_id).where(
            EventEntity.entity_id == entity_id
        )
        entity_event_ids: list[uuid.UUID] = list(
            (await pg_session.execute(entity_event_ids_stmt)).scalars().all()
        )

        if entity_event_ids:
            # Fetch those events and score similarity in Python since we
            # can't apply entity filter directly inside similarity_search.
            from sqlalchemy import func
            from app.models.sql.event import Event as _Event

            dist_expr = _Event.embedding.cosine_distance(question_embedding)
            stmt = (
                select(_Event, dist_expr.label("distance"))
                .where(_Event.id.in_(entity_event_ids))
                .where(_Event.embedding.is_not(None))
                .order_by(dist_expr.asc())
                .limit(_SEED_LIMIT)
            )
            rows = (await pg_session.execute(stmt)).all()
            seed_events: list[Event] = [row.Event for row in rows]
        else:
            seed_events = []
    else:
        pairs = await event_store.similarity_search(
            pg_session,
            question_embedding,
            limit=_SEED_LIMIT,
            max_distance=_SEED_MAX_DISTANCE,
        )
        seed_events = [event for event, _dist in pairs]

    if not seed_events:
        logger.info("causal_planner_no_seeds", question=question)
        return PlanResult(confidence=0.0)

    # ── Step 2: traverse causal graph from each seed ───────────────────────
    all_chain_records: list[dict[str, Any]] = []
    seen_chain_ids: set[str] = set()

    for seed in seed_events:
        chain = await graph_store.get_causal_chain(
            neo4j_session,
            seed.id,
            direction="both",
            max_hops=max_hops,
        )
        for record in chain:
            eid = record.get("event_id")
            if eid and eid not in seen_chain_ids:
                seen_chain_ids.add(eid)
                all_chain_records.append(record)

    # Sort by hop distance so the synthesizer sees nearest nodes first.
    all_chain_records.sort(key=lambda r: r.get("hop", 0))

    # ── Step 3: fetch full event details from PostgreSQL ───────────────────
    full_event_ids: list[uuid.UUID] = []
    for record in all_chain_records:
        try:
            full_event_ids.append(uuid.UUID(record["event_id"]))
        except (KeyError, ValueError):
            continue

    # Also include the seed events themselves.
    seed_ids = {e.id for e in seed_events}
    extra_ids = [eid for eid in full_event_ids if eid not in seed_ids]

    fetched: list[Event] = list(seed_events)
    if extra_ids:
        stmt = select(Event).where(Event.id.in_(extra_ids))
        fetched += list((await pg_session.execute(stmt)).scalars().all())

    # Deduplicate preserving order.
    seen_event_ids: set[uuid.UUID] = set()
    unique_events: list[Event] = []
    for ev in fetched:
        if ev.id not in seen_event_ids:
            seen_event_ids.add(ev.id)
            unique_events.append(ev)

    document_ids = {ev.document_id for ev in unique_events}

    # Confidence is the mean seed similarity confidence (proxy).
    confidence = min(0.90, 0.70 + 0.10 * len(all_chain_records))

    logger.info(
        "causal_planner_complete",
        seeds=len(seed_events),
        chain_nodes=len(all_chain_records),
        events=len(unique_events),
    )
    return PlanResult(
        events=unique_events,
        causal_chain=all_chain_records,
        document_ids=document_ids,
        confidence=confidence,
    )
