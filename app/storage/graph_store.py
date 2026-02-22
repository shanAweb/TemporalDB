"""
app/storage/graph_store.py

Neo4j CRUD for the causal event graph.

Node labels
-----------
  :Event   — mirrors the PostgreSQL events table.
  :Entity  — mirrors the PostgreSQL entities table.

Relationship types
------------------
  (:Event)-[:CAUSES {confidence, evidence, relation_id}]->(:Event)
  (:Event)-[:INVOLVES]->(:Entity)

All write functions use MERGE so they are idempotent (safe to call more
than once with the same IDs, e.g. during re-processing).

The Neo4j AsyncSession is obtained from app.database.neo4j.get_neo4j()
and should be passed in by the caller.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import structlog
from neo4j import AsyncSession

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _str(value: uuid.UUID | None) -> str | None:
    """Convert UUID to string for Neo4j (Neo4j has no native UUID type)."""
    return str(value) if value is not None else None


def _iso(value: datetime | None) -> str | None:
    """Convert datetime to ISO-8601 string for Neo4j storage."""
    return value.isoformat() if value is not None else None


# ---------------------------------------------------------------------------
# Event node CRUD
# ---------------------------------------------------------------------------

async def upsert_event_node(
    session: AsyncSession,
    *,
    event_id: uuid.UUID,
    description: str,
    event_type: str | None = None,
    ts_start: datetime | None = None,
    ts_end: datetime | None = None,
    confidence: float = 1.0,
    source_sentence: str | None = None,
    document_id: uuid.UUID | None = None,
) -> None:
    """Create or update an :Event node in Neo4j.

    Uses MERGE on the ``id`` property so repeated calls are idempotent.
    All other properties are set via SET … on the matched or newly-created node.

    Args:
        session:         Neo4j async session.
        event_id:        UUID of the event (primary key from PostgreSQL).
        description:     Human-readable event description.
        event_type:      Optional event type label (action, state_change, …).
        ts_start:        Event start timestamp (UTC).
        ts_end:          Event end timestamp (UTC), or None for point-in-time.
        confidence:      NLP extraction confidence score [0.0 – 1.0].
        source_sentence: Original sentence the event was extracted from.
        document_id:     UUID of the source document.
    """
    cypher = """
        MERGE (e:Event {id: $id})
        SET
            e.description     = $description,
            e.event_type      = $event_type,
            e.ts_start        = $ts_start,
            e.ts_end          = $ts_end,
            e.confidence      = $confidence,
            e.source_sentence = $source_sentence,
            e.document_id     = $document_id
    """
    params: dict[str, Any] = {
        "id":              _str(event_id),
        "description":     description,
        "event_type":      event_type,
        "ts_start":        _iso(ts_start),
        "ts_end":          _iso(ts_end),
        "confidence":      confidence,
        "source_sentence": source_sentence,
        "document_id":     _str(document_id),
    }
    await session.run(cypher, params)
    logger.debug("graph_event_upserted", event_id=str(event_id))


async def delete_event_node(
    session: AsyncSession,
    event_id: uuid.UUID,
) -> bool:
    """Delete an :Event node and all its relationships.

    Returns True if the node existed and was deleted, False if not found.

    Args:
        session:  Neo4j async session.
        event_id: UUID of the event to delete.
    """
    cypher = """
        MATCH (e:Event {id: $id})
        DETACH DELETE e
        RETURN count(e) AS deleted
    """
    result = await session.run(cypher, {"id": _str(event_id)})
    record = await result.single()
    deleted = bool(record and record["deleted"] > 0)
    logger.debug("graph_event_deleted", event_id=str(event_id), found=deleted)
    return deleted


# ---------------------------------------------------------------------------
# Entity node CRUD
# ---------------------------------------------------------------------------

async def upsert_entity_node(
    session: AsyncSession,
    *,
    entity_id: uuid.UUID,
    name: str,
    canonical_name: str,
    entity_type: str,
) -> None:
    """Create or update an :Entity node in Neo4j.

    Uses MERGE on the ``id`` property so repeated calls are idempotent.

    Args:
        session:        Neo4j async session.
        entity_id:      UUID of the entity (primary key from PostgreSQL).
        name:           Raw mention text.
        canonical_name: Resolved canonical name for cross-document linking.
        entity_type:    spaCy NER label (PERSON, ORG, GPE, …).
    """
    cypher = """
        MERGE (en:Entity {id: $id})
        SET
            en.name           = $name,
            en.canonical_name = $canonical_name,
            en.type           = $type
    """
    params: dict[str, Any] = {
        "id":             _str(entity_id),
        "name":           name,
        "canonical_name": canonical_name,
        "type":           entity_type,
    }
    await session.run(cypher, params)
    logger.debug("graph_entity_upserted", entity_id=str(entity_id))


# ---------------------------------------------------------------------------
# Relationship CRUD
# ---------------------------------------------------------------------------

async def upsert_causal_edge(
    session: AsyncSession,
    *,
    cause_event_id: uuid.UUID,
    effect_event_id: uuid.UUID,
    relation_id: uuid.UUID,
    confidence: float = 1.0,
    evidence: str | None = None,
) -> None:
    """Create or update a CAUSES relationship between two :Event nodes.

    MERGE is keyed on the ``relation_id`` property so the relationship is
    idempotent even if the same pair appears in multiple documents.

    Both event nodes must already exist (call upsert_event_node first).

    Args:
        session:         Neo4j async session.
        cause_event_id:  UUID of the causing event.
        effect_event_id: UUID of the effect event.
        relation_id:     UUID of the CausalRelation row in PostgreSQL.
        confidence:      Confidence score for this causal link [0.0 – 1.0].
        evidence:        Source phrase or sentence that signals the causal link.
    """
    cypher = """
        MATCH (cause:Event {id: $cause_id})
        MATCH (effect:Event {id: $effect_id})
        MERGE (cause)-[r:CAUSES {relation_id: $relation_id}]->(effect)
        SET
            r.confidence = $confidence,
            r.evidence   = $evidence
    """
    params: dict[str, Any] = {
        "cause_id":    _str(cause_event_id),
        "effect_id":   _str(effect_event_id),
        "relation_id": _str(relation_id),
        "confidence":  confidence,
        "evidence":    evidence,
    }
    await session.run(cypher, params)
    logger.debug(
        "graph_causal_edge_upserted",
        cause=str(cause_event_id),
        effect=str(effect_event_id),
    )


async def upsert_involves_edge(
    session: AsyncSession,
    *,
    event_id: uuid.UUID,
    entity_id: uuid.UUID,
) -> None:
    """Create or update an INVOLVES relationship between an :Event and :Entity.

    Both nodes must already exist.

    Args:
        session:   Neo4j async session.
        event_id:  UUID of the event.
        entity_id: UUID of the entity.
    """
    cypher = """
        MATCH (e:Event  {id: $event_id})
        MATCH (en:Entity {id: $entity_id})
        MERGE (e)-[:INVOLVES]->(en)
    """
    params: dict[str, Any] = {
        "event_id":  _str(event_id),
        "entity_id": _str(entity_id),
    }
    await session.run(cypher, params)
    logger.debug(
        "graph_involves_edge_upserted",
        event_id=str(event_id),
        entity_id=str(entity_id),
    )


async def delete_causal_edge(
    session: AsyncSession,
    relation_id: uuid.UUID,
) -> bool:
    """Delete a CAUSES relationship by its relation_id property.

    Returns True if the relationship existed and was deleted, False otherwise.

    Args:
        session:     Neo4j async session.
        relation_id: UUID of the CausalRelation row in PostgreSQL.
    """
    cypher = """
        MATCH ()-[r:CAUSES {relation_id: $relation_id}]->()
        DELETE r
        RETURN count(r) AS deleted
    """
    result = await session.run(cypher, {"relation_id": _str(relation_id)})
    record = await result.single()
    deleted = bool(record and record["deleted"] > 0)
    logger.debug("graph_causal_edge_deleted", relation_id=str(relation_id), found=deleted)
    return deleted


# ---------------------------------------------------------------------------
# Causal chain traversal
# ---------------------------------------------------------------------------

async def get_causal_chain(
    session: AsyncSession,
    event_id: uuid.UUID,
    *,
    direction: str = "downstream",
    max_hops: int = 5,
) -> list[dict[str, Any]]:
    """Traverse the causal graph from a seed event and return the chain.

    Uses variable-length path matching (up to *max_hops* hops) to find all
    causally connected events.

    Args:
        session:   Neo4j async session.
        event_id:  Starting event UUID.
        direction: ``"downstream"`` follows CAUSES edges forward (what did
                   this event cause?); ``"upstream"`` follows them backward
                   (what caused this event?); ``"both"`` traverses in both
                   directions simultaneously.
        max_hops:  Maximum number of relationship hops to traverse (1–10).

    Returns:
        List of dicts, each representing a node in the chain::

            {
                "event_id":   str,
                "description": str,
                "event_type":  str | None,
                "ts_start":    str | None,
                "confidence":  float,
                "hop":         int,         # distance from seed
                "relation_confidence": float | None,  # edge confidence
                "evidence":    str | None,
            }
    """
    max_hops = max(1, min(max_hops, 10))  # clamp to [1, 10]

    if direction == "downstream":
        path_clause = f"(seed)-[:CAUSES*1..{max_hops}]->(related)"
        rel_clause  = f"(seed)-[r:CAUSES*1..{max_hops}]->(related)"
    elif direction == "upstream":
        path_clause = f"(seed)<-[:CAUSES*1..{max_hops}]-(related)"
        rel_clause  = f"(seed)<-[r:CAUSES*1..{max_hops}]-(related)"
    else:  # both
        path_clause = f"(seed)-[:CAUSES*1..{max_hops}]-(related)"
        rel_clause  = f"(seed)-[r:CAUSES*1..{max_hops}]-(related)"

    # Two-query approach: first get reachable nodes + hop distance,
    # then fetch the last-hop edge properties for confidence + evidence.
    cypher = f"""
        MATCH (seed:Event {{id: $id}})
        MATCH p = {path_clause}
        WITH related, length(p) AS hop
        ORDER BY hop ASC
        RETURN DISTINCT
            related.id            AS event_id,
            related.description   AS description,
            related.event_type    AS event_type,
            related.ts_start      AS ts_start,
            related.confidence    AS confidence,
            hop
    """
    result = await session.run(cypher, {"id": _str(event_id)})
    records = await result.data()

    logger.debug(
        "graph_causal_chain_fetched",
        seed=str(event_id),
        direction=direction,
        nodes=len(records),
    )
    return records


# ---------------------------------------------------------------------------
# Entity-centric queries
# ---------------------------------------------------------------------------

async def get_entity_graph(
    session: AsyncSession,
    entity_id: uuid.UUID,
    *,
    max_events: int = 50,
) -> dict[str, Any]:
    """Return all events involving an entity and the causal edges among them.

    The result contains two lists suitable for rendering a graph or building
    a timeline:

    - ``events``: every :Event node that has an INVOLVES edge to the entity.
    - ``edges``:  every CAUSES relationship among those events.

    Args:
        session:    Neo4j async session.
        entity_id:  Entity UUID to centre the subgraph on.
        max_events: Maximum number of event nodes to return (ordered by
                    ts_start ascending, NULLs last).

    Returns:
        ``{"events": [...], "edges": [...]}``
        where each event dict has keys: event_id, description, event_type,
        ts_start, confidence; and each edge dict has: cause_id, effect_id,
        relation_id, confidence, evidence.
    """
    event_cypher = """
        MATCH (en:Entity {id: $entity_id})<-[:INVOLVES]-(e:Event)
        RETURN
            e.id            AS event_id,
            e.description   AS description,
            e.event_type    AS event_type,
            e.ts_start      AS ts_start,
            e.confidence    AS confidence
        ORDER BY
            CASE WHEN e.ts_start IS NULL THEN 1 ELSE 0 END ASC,
            e.ts_start ASC
        LIMIT $limit
    """
    event_result = await session.run(
        event_cypher, {"entity_id": _str(entity_id), "limit": max_events}
    )
    events: list[dict[str, Any]] = await event_result.data()

    if not events:
        return {"events": [], "edges": []}

    event_ids = [e["event_id"] for e in events]

    edge_cypher = """
        MATCH (cause:Event)-[r:CAUSES]->(effect:Event)
        WHERE cause.id IN $ids AND effect.id IN $ids
        RETURN
            cause.id    AS cause_id,
            effect.id   AS effect_id,
            r.relation_id AS relation_id,
            r.confidence  AS confidence,
            r.evidence    AS evidence
    """
    edge_result = await session.run(edge_cypher, {"ids": event_ids})
    edges: list[dict[str, Any]] = await edge_result.data()

    logger.debug(
        "graph_entity_graph_fetched",
        entity_id=str(entity_id),
        events=len(events),
        edges=len(edges),
    )
    return {"events": events, "edges": edges}


async def get_causal_path_between(
    session: AsyncSession,
    source_event_id: uuid.UUID,
    target_event_id: uuid.UUID,
    *,
    max_hops: int = 5,
) -> list[dict[str, Any]]:
    """Find the shortest causal path between two events.

    Uses Neo4j's shortestPath algorithm over CAUSES edges.

    Args:
        session:         Neo4j async session.
        source_event_id: UUID of the start event.
        target_event_id: UUID of the end event.
        max_hops:        Maximum path length to search (clamped to [1, 10]).

    Returns:
        Ordered list of event dicts along the shortest path (empty if none
        exists within the hop limit). Each dict has: event_id, description,
        event_type, ts_start, confidence.
    """
    max_hops = max(1, min(max_hops, 10))

    cypher = f"""
        MATCH (src:Event {{id: $source_id}}), (tgt:Event {{id: $target_id}})
        MATCH p = shortestPath((src)-[:CAUSES*1..{max_hops}]->(tgt))
        UNWIND nodes(p) AS n
        RETURN
            n.id           AS event_id,
            n.description  AS description,
            n.event_type   AS event_type,
            n.ts_start     AS ts_start,
            n.confidence   AS confidence
    """
    result = await session.run(
        cypher,
        {
            "source_id": _str(source_event_id),
            "target_id": _str(target_event_id),
        },
    )
    records: list[dict[str, Any]] = await result.data()
    logger.debug(
        "graph_path_fetched",
        source=str(source_event_id),
        target=str(target_event_id),
        hops=len(records) - 1 if records else 0,
    )
    return records
