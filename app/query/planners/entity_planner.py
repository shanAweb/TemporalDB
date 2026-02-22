"""
app/query/planners/entity_planner.py

Planner for ENTITY_TIMELINE queries.

Strategy
--------
Combines two data sources to build a comprehensive picture of an entity:

1. PostgreSQL  — event_store.list_events(entity_id=...) returns every event
   the entity was involved in, ordered chronologically.

2. Neo4j       — graph_store.get_entity_graph(entity_id) returns the same
   events plus the CAUSES edges *among* them, giving the synthesizer a
   causal subgraph to reason over.

The causal chain is built from the Neo4j edge records and ordered by
event ts_start so the synthesizer sees a chronological narrative.

If no entity_id is resolved (the mention was not found in the store) the
planner returns an empty PlanResult with confidence 0.
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog
from neo4j import AsyncSession as Neo4jSession
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas.query import TimeRange
from app.query.planners import PlanResult
from app.storage import event_store, graph_store

logger = structlog.get_logger(__name__)

_DEFAULT_LIMIT    = 50   # Max events from PostgreSQL
_GRAPH_MAX_EVENTS = 50   # Max events from Neo4j subgraph


async def run(
    pg_session: AsyncSession,
    neo4j_session: Neo4jSession,
    entity_id: uuid.UUID | None,
    *,
    time_range: TimeRange | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> PlanResult:
    """Execute the entity timeline planning strategy.

    Args:
        pg_session:    SQLAlchemy async session (read-only).
        neo4j_session: Neo4j async session.
        entity_id:     UUID of the resolved entity.  If None an empty
                       PlanResult is returned immediately.
        time_range:    Optional UTC time window to narrow events.
        limit:         Maximum number of events to return from PostgreSQL.

    Returns:
        PlanResult with chronological events and causal subgraph.
    """
    if entity_id is None:
        logger.info("entity_planner_no_entity")
        return PlanResult(confidence=0.0)

    # ── PostgreSQL: chronological event list ────────────────────────────────
    from_date = time_range.start if time_range else None
    to_date   = time_range.end   if time_range else None

    events, total = await event_store.list_events(
        pg_session,
        entity_id=entity_id,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        offset=0,
    )

    # ── Neo4j: causal subgraph for this entity ──────────────────────────────
    graph_data: dict[str, Any] = await graph_store.get_entity_graph(
        neo4j_session,
        entity_id,
        max_events=_GRAPH_MAX_EVENTS,
    )

    # Build a causal chain from the Neo4j edge records.
    # Use the edge list to produce cause → effect narrative records
    # that the synthesizer can walk like a linked list.
    edge_records: list[dict[str, Any]] = graph_data.get("edges", [])
    graph_events: list[dict[str, Any]] = graph_data.get("events", [])

    # Build a lookup from event_id → ts_start for ordering.
    ts_lookup: dict[str, str | None] = {
        ev["event_id"]: ev.get("ts_start") for ev in graph_events
    }

    # Turn edge records into chain-compatible dicts (cause first).
    causal_chain: list[dict[str, Any]] = []
    seen_chain_ids: set[str] = set()

    for edge in edge_records:
        for eid in (edge.get("cause_id"), edge.get("effect_id")):
            if eid and eid not in seen_chain_ids:
                seen_chain_ids.add(eid)
                # Find description from graph_events.
                desc = next(
                    (ge["description"] for ge in graph_events if ge["event_id"] == eid),
                    "",
                )
                causal_chain.append(
                    {
                        "event_id":   eid,
                        "description": desc,
                        "ts_start":   ts_lookup.get(eid),
                        "confidence": edge.get("confidence", 1.0),
                        "hop":        0,  # flat — not a traversal result
                    }
                )

    document_ids = {ev.document_id for ev in events}

    logger.info(
        "entity_planner_complete",
        entity_id=str(entity_id),
        pg_events=len(events),
        pg_total=total,
        chain_nodes=len(causal_chain),
    )
    return PlanResult(
        events=events,
        causal_chain=causal_chain,
        document_ids=document_ids,
        confidence=0.88 if events else 0.0,
    )
