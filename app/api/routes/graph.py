"""
app/api/routes/graph.py

Graph visualisation endpoint.

GET /graph/entity/{entity_id}
    Return the causal subgraph centred on an entity as a list of nodes
    and edges, suitable for rendering in a graph visualisation library
    (e.g. D3, Cytoscape, Sigma).

    Nodes
    -----
      type="entity"  — the anchor entity (one node).
      type="event"   — every event that INVOLVES the entity.

    Edges
    -----
      type="INVOLVES" — one edge from each event node to the entity node.
      type="CAUSES"   — causal edges between event nodes.
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from neo4j import AsyncSession as Neo4jSession
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import require_api_key
from app.database.neo4j import get_neo4j
from app.database.postgres import get_db
from app.models.schemas.graph import GraphEdge, GraphNode, GraphResponse
from app.storage import entity_store, graph_store

logger = structlog.get_logger(__name__)

router = APIRouter()


def _parse_uuid(value: str | None) -> uuid.UUID | None:
    """Safely parse a string UUID; return None on failure."""
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def _build_graph(
    entity_id: uuid.UUID,
    entity_name: str,
    graph_data: dict[str, Any],
) -> GraphResponse:
    """Convert raw Neo4j graph data into a GraphResponse.

    Args:
        entity_id:  UUID of the anchor entity.
        entity_name: Canonical name for the entity node label.
        graph_data: Dict with "events" and "edges" lists from graph_store.

    Returns:
        GraphResponse with typed nodes and edges.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # ── Entity node (anchor) ──────────────────────────────────────────────
    nodes.append(
        GraphNode(
            id=entity_id,
            label=entity_name,
            type="entity",
            properties={},
        )
    )

    # ── Event nodes + INVOLVES edges ──────────────────────────────────────
    for ev in graph_data.get("events", []):
        ev_uuid = _parse_uuid(ev.get("event_id"))
        if ev_uuid is None:
            continue

        props: dict[str, Any] = {}
        if ev.get("event_type"):
            props["event_type"] = ev["event_type"]
        if ev.get("ts_start"):
            props["ts_start"] = ev["ts_start"]
        if ev.get("confidence") is not None:
            props["confidence"] = ev["confidence"]

        nodes.append(
            GraphNode(
                id=ev_uuid,
                label=ev.get("description", ""),
                type="event",
                properties=props,
            )
        )

        edges.append(
            GraphEdge(
                source=ev_uuid,
                target=entity_id,
                type="INVOLVES",
                confidence=None,
            )
        )

    # ── CAUSES edges among event nodes ────────────────────────────────────
    for edge in graph_data.get("edges", []):
        cause_uuid  = _parse_uuid(edge.get("cause_id"))
        effect_uuid = _parse_uuid(edge.get("effect_id"))
        if cause_uuid is None or effect_uuid is None:
            continue

        edges.append(
            GraphEdge(
                source=cause_uuid,
                target=effect_uuid,
                type="CAUSES",
                confidence=edge.get("confidence"),
            )
        )

    return GraphResponse(nodes=nodes, edges=edges)


@router.get(
    "/entity/{entity_id}",
    response_model=GraphResponse,
    summary="Get causal subgraph for an entity",
)
async def get_entity_graph(
    entity_id: uuid.UUID,
    max_events: int = Query(default=50, ge=1, le=200, description="Maximum event nodes to include"),
    pg_session: AsyncSession = Depends(get_db),
    neo4j_session: Neo4jSession = Depends(get_neo4j),
    _key: str = Depends(require_api_key),
) -> GraphResponse:
    """Return the causal subgraph centred on *entity_id*.

    Includes:
    - One entity node (the anchor).
    - Up to *max_events* event nodes ordered chronologically.
    - INVOLVES edges from each event to the entity.
    - CAUSES edges between event nodes.

    Returns HTTP 404 when the entity does not exist in PostgreSQL.
    """
    # Verify entity exists in PostgreSQL for a clean 404 experience.
    entity = await entity_store.get_entity_by_id(pg_session, entity_id)
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found.",
        )

    graph_data = await graph_store.get_entity_graph(
        neo4j_session,
        entity_id,
        max_events=max_events,
    )

    response = _build_graph(entity_id, entity.canonical_name, graph_data)

    logger.info(
        "graph_entity_served",
        entity_id=str(entity_id),
        nodes=len(response.nodes),
        edges=len(response.edges),
    )
    return response
