"""
tests/unit/test_entity_planner.py

Unit tests for app.query.planners.entity_planner.

event_store.list_events and graph_store.get_entity_graph are mocked so
no database or Neo4j instance is required.

Coverage
--------
  - run() with entity_id=None → empty PlanResult, confidence 0.0
  - run() with entity_id → list_events called with entity_id
  - run() collects document_ids from PG events
  - run() with events → confidence 0.88
  - run() with no events → confidence 0.0
  - causal_chain built from Neo4j edge records (cause and effect IDs)
  - duplicate event IDs in chain deduplicated (seen_chain_ids)
  - causal_chain empty when graph has no edges
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.query.planners import PlanResult
from app.query.planners import entity_planner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sessions() -> tuple[AsyncMock, AsyncMock]:
    return AsyncMock(), AsyncMock()


def _mock_event(doc_id: uuid.UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        description="Revenue fell.",
        document_id=doc_id or uuid.uuid4(),
        ts_start=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )


def _graph_data(
    event_id: str | None = None,
    cause_id: str | None = None,
    effect_id: str | None = None,
) -> dict:
    events = []
    edges  = []
    if event_id:
        events.append({
            "event_id":    event_id,
            "description": "Costs rose.",
            "ts_start":    "2024-07-01T00:00:00",
        })
    if cause_id and effect_id:
        edges.append({
            "cause_id":  cause_id,
            "effect_id": effect_id,
            "confidence": 0.80,
        })
    return {"events": events, "edges": edges}


async def _run(
    entity_id: uuid.UUID | None,
    pg_events: list | None = None,
    graph: dict | None = None,
) -> PlanResult:
    pg, neo4j = _sessions()
    pg_events = pg_events or []
    graph     = graph or {"events": [], "edges": []}

    with (
        patch("app.query.planners.entity_planner.event_store.list_events",
              return_value=(pg_events, len(pg_events))),
        patch("app.query.planners.entity_planner.graph_store.get_entity_graph",
              return_value=graph),
    ):
        return await entity_planner.run(pg, neo4j, entity_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEntityPlannerNoEntity:
    @pytest.mark.asyncio
    async def test_none_entity_id_returns_empty_plan(self) -> None:
        result = await _run(entity_id=None)
        assert result.events == []
        assert result.causal_chain == []

    @pytest.mark.asyncio
    async def test_none_entity_id_confidence_zero(self) -> None:
        result = await _run(entity_id=None)
        assert result.confidence == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_none_entity_id_skips_store_calls(self) -> None:
        pg, neo4j = _sessions()
        with (
            patch("app.query.planners.entity_planner.event_store.list_events") as mock_list,
            patch("app.query.planners.entity_planner.graph_store.get_entity_graph") as mock_graph,
        ):
            await entity_planner.run(pg, neo4j, None)
        mock_list.assert_not_awaited()
        mock_graph.assert_not_awaited()


class TestEntityPlannerWithEntity:
    @pytest.mark.asyncio
    async def test_list_events_called_with_entity_id(self) -> None:
        entity_id = uuid.uuid4()
        pg, neo4j = _sessions()
        with (
            patch("app.query.planners.entity_planner.event_store.list_events",
                  return_value=([], 0)) as mock_list,
            patch("app.query.planners.entity_planner.graph_store.get_entity_graph",
                  return_value={"events": [], "edges": []}),
        ):
            await entity_planner.run(pg, neo4j, entity_id)
        assert mock_list.call_args.kwargs["entity_id"] == entity_id

    @pytest.mark.asyncio
    async def test_events_populated_from_pg(self) -> None:
        ev = _mock_event()
        result = await _run(uuid.uuid4(), pg_events=[ev])
        assert result.events == [ev]

    @pytest.mark.asyncio
    async def test_document_ids_from_pg_events(self) -> None:
        doc_id = uuid.uuid4()
        ev = _mock_event(doc_id=doc_id)
        result = await _run(uuid.uuid4(), pg_events=[ev])
        assert doc_id in result.document_ids

    @pytest.mark.asyncio
    async def test_confidence_088_when_events_found(self) -> None:
        ev = _mock_event()
        result = await _run(uuid.uuid4(), pg_events=[ev])
        assert result.confidence == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_confidence_zero_when_no_events(self) -> None:
        result = await _run(uuid.uuid4(), pg_events=[])
        assert result.confidence == pytest.approx(0.0)


class TestEntityPlannerCausalChain:
    @pytest.mark.asyncio
    async def test_empty_edges_produce_empty_chain(self) -> None:
        ev = _mock_event()
        result = await _run(uuid.uuid4(), pg_events=[ev], graph={"events": [], "edges": []})
        assert result.causal_chain == []

    @pytest.mark.asyncio
    async def test_edge_adds_cause_and_effect_to_chain(self) -> None:
        cause_id  = str(uuid.uuid4())
        effect_id = str(uuid.uuid4())
        graph = _graph_data(
            event_id=cause_id,
            cause_id=cause_id,
            effect_id=effect_id,
        )
        ev = _mock_event()
        result = await _run(uuid.uuid4(), pg_events=[ev], graph=graph)
        chain_ids = {r["event_id"] for r in result.causal_chain}
        assert cause_id  in chain_ids
        assert effect_id in chain_ids

    @pytest.mark.asyncio
    async def test_duplicate_chain_ids_deduplicated(self) -> None:
        event_id = str(uuid.uuid4())
        # Two edges sharing the same cause_id — should appear once in chain.
        graph = {
            "events": [{"event_id": event_id, "description": "X", "ts_start": None}],
            "edges": [
                {"cause_id": event_id, "effect_id": str(uuid.uuid4()), "confidence": 0.8},
                {"cause_id": event_id, "effect_id": str(uuid.uuid4()), "confidence": 0.8},
            ],
        }
        ev = _mock_event()
        result = await _run(uuid.uuid4(), pg_events=[ev], graph=graph)
        cause_count = sum(1 for r in result.causal_chain if r["event_id"] == event_id)
        assert cause_count == 1
