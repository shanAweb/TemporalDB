"""
tests/unit/test_sync.py

Unit tests for app.storage.sync.

Tests the SyncResult dataclass and the sync_document() function with
mocked PostgreSQL and Neo4j sessions.  All graph_store calls are patched
so no running Neo4j is required.

Coverage
--------
  - SyncResult.__add__: combines all four counter fields correctly
  - SyncResult.as_dict: returns the expected dict keys and values
  - sync_document: no events → returns zero-count SyncResult immediately
  - sync_document: syncs event nodes, entity nodes, INVOLVES, CAUSES edges
  - sync_document: entity deduplication across events (unique set only)
  - sync_document: graph_store functions called correct number of times
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.storage.sync import SyncResult, sync_document


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------

class TestSyncResult:
    def test_add_combines_counters(self) -> None:
        a = SyncResult(event_nodes=1, entity_nodes=2, involves_edges=3, causal_edges=4)
        b = SyncResult(event_nodes=10, entity_nodes=20, involves_edges=30, causal_edges=40)
        c = a + b
        assert c.event_nodes    == 11
        assert c.entity_nodes   == 22
        assert c.involves_edges == 33
        assert c.causal_edges   == 44

    def test_add_identity(self) -> None:
        a = SyncResult(event_nodes=5, entity_nodes=3, involves_edges=7, causal_edges=2)
        assert (a + SyncResult()) == a

    def test_as_dict_keys(self) -> None:
        result = SyncResult(event_nodes=1, entity_nodes=2, involves_edges=3, causal_edges=4)
        d = result.as_dict()
        assert set(d.keys()) == {"event_nodes", "entity_nodes", "involves_edges", "causal_edges"}

    def test_as_dict_values(self) -> None:
        result = SyncResult(event_nodes=1, entity_nodes=2, involves_edges=3, causal_edges=4)
        d = result.as_dict()
        assert d["event_nodes"]    == 1
        assert d["entity_nodes"]   == 2
        assert d["involves_edges"] == 3
        assert d["causal_edges"]   == 4

    def test_default_values_are_zero(self) -> None:
        r = SyncResult()
        assert r.event_nodes == r.entity_nodes == r.involves_edges == r.causal_edges == 0


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------

def _scalars_all(rows: list) -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.all.return_value = rows
    return r


def _mock_entity(entity_id: uuid.UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=entity_id or uuid.uuid4(),
        name="Acme Corp",
        canonical_name="acme corp",
        type="ORG",
    )


def _mock_event(
    doc_id: uuid.UUID,
    entities: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        description="Revenue fell.",
        event_type="state_change",
        ts_start=None,
        ts_end=None,
        confidence=0.85,
        source_sentence=None,
        document_id=doc_id,
        entities=entities or [],
    )


def _mock_involves_link(event_id: uuid.UUID, entity_id: uuid.UUID) -> SimpleNamespace:
    return SimpleNamespace(event_id=event_id, entity_id=entity_id)


def _mock_causal_relation(
    cause_id: uuid.UUID,
    effect_id: uuid.UUID,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        cause_event_id=cause_id,
        effect_event_id=effect_id,
        confidence=0.80,
        evidence=None,
    )


# ---------------------------------------------------------------------------
# sync_document
# ---------------------------------------------------------------------------

class TestSyncDocumentNoEvents:
    @pytest.mark.asyncio
    async def test_returns_zero_counts_when_no_events(self) -> None:
        pg = AsyncMock()
        neo4j = AsyncMock()
        # First execute returns no events.
        pg.execute.return_value = _scalars_all([])

        result = await sync_document(pg, neo4j, uuid.uuid4())

        assert result.event_nodes    == 0
        assert result.entity_nodes   == 0
        assert result.involves_edges == 0
        assert result.causal_edges   == 0

    @pytest.mark.asyncio
    async def test_only_one_pg_query_when_no_events(self) -> None:
        pg = AsyncMock()
        pg.execute.return_value = _scalars_all([])
        await sync_document(pg, AsyncMock(), uuid.uuid4())
        # Only the initial event query should be issued.
        assert pg.execute.await_count == 1


class TestSyncDocumentWithEvents:
    @pytest.mark.asyncio
    async def test_event_nodes_count(self) -> None:
        doc_id = uuid.uuid4()
        event  = _mock_event(doc_id)
        pg     = AsyncMock()
        # execute calls: events, involves links, causal relations.
        pg.execute.side_effect = [
            _scalars_all([event]),   # events
            _scalars_all([]),        # involves
            _scalars_all([]),        # causal
        ]
        with (
            patch("app.storage.sync.graph_store.upsert_event_node"),
            patch("app.storage.sync.graph_store.upsert_entity_node"),
            patch("app.storage.sync.graph_store.upsert_involves_edge"),
            patch("app.storage.sync.graph_store.upsert_causal_edge"),
        ):
            result = await sync_document(pg, AsyncMock(), doc_id)

        assert result.event_nodes == 1

    @pytest.mark.asyncio
    async def test_entity_nodes_counted(self) -> None:
        doc_id = uuid.uuid4()
        entity = _mock_entity()
        event  = _mock_event(doc_id, entities=[entity])
        pg     = AsyncMock()
        pg.execute.side_effect = [
            _scalars_all([event]),
            _scalars_all([]),        # involves
            _scalars_all([]),        # causal
        ]
        with (
            patch("app.storage.sync.graph_store.upsert_event_node"),
            patch("app.storage.sync.graph_store.upsert_entity_node"),
            patch("app.storage.sync.graph_store.upsert_involves_edge"),
            patch("app.storage.sync.graph_store.upsert_causal_edge"),
        ):
            result = await sync_document(pg, AsyncMock(), doc_id)

        assert result.entity_nodes == 1

    @pytest.mark.asyncio
    async def test_entity_deduplication_across_events(self) -> None:
        """Same entity linked to two events → only one entity node synced."""
        doc_id = uuid.uuid4()
        entity = _mock_entity()
        ev1    = _mock_event(doc_id, entities=[entity])
        ev2    = _mock_event(doc_id, entities=[entity])
        pg     = AsyncMock()
        pg.execute.side_effect = [
            _scalars_all([ev1, ev2]),
            _scalars_all([]),
            _scalars_all([]),
        ]
        upsert_entity_mock = AsyncMock()
        with (
            patch("app.storage.sync.graph_store.upsert_event_node"),
            patch("app.storage.sync.graph_store.upsert_entity_node", upsert_entity_mock),
            patch("app.storage.sync.graph_store.upsert_involves_edge"),
            patch("app.storage.sync.graph_store.upsert_causal_edge"),
        ):
            result = await sync_document(pg, AsyncMock(), doc_id)

        # Entity appears in both events but should be synced only once.
        assert result.entity_nodes == 1
        upsert_entity_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_involves_edges_counted(self) -> None:
        doc_id = uuid.uuid4()
        event  = _mock_event(doc_id)
        link   = _mock_involves_link(event.id, uuid.uuid4())
        pg     = AsyncMock()
        pg.execute.side_effect = [
            _scalars_all([event]),
            _scalars_all([link]),
            _scalars_all([]),
        ]
        with (
            patch("app.storage.sync.graph_store.upsert_event_node"),
            patch("app.storage.sync.graph_store.upsert_entity_node"),
            patch("app.storage.sync.graph_store.upsert_involves_edge"),
            patch("app.storage.sync.graph_store.upsert_causal_edge"),
        ):
            result = await sync_document(pg, AsyncMock(), doc_id)

        assert result.involves_edges == 1

    @pytest.mark.asyncio
    async def test_causal_edges_counted(self) -> None:
        doc_id = uuid.uuid4()
        ev1    = _mock_event(doc_id)
        ev2    = _mock_event(doc_id)
        rel    = _mock_causal_relation(ev1.id, ev2.id)
        pg     = AsyncMock()
        pg.execute.side_effect = [
            _scalars_all([ev1, ev2]),
            _scalars_all([]),
            _scalars_all([rel]),
        ]
        with (
            patch("app.storage.sync.graph_store.upsert_event_node"),
            patch("app.storage.sync.graph_store.upsert_entity_node"),
            patch("app.storage.sync.graph_store.upsert_involves_edge"),
            patch("app.storage.sync.graph_store.upsert_causal_edge"),
        ):
            result = await sync_document(pg, AsyncMock(), doc_id)

        assert result.causal_edges == 1

    @pytest.mark.asyncio
    async def test_graph_store_upsert_event_called_per_event(self) -> None:
        doc_id      = uuid.uuid4()
        events      = [_mock_event(doc_id), _mock_event(doc_id)]
        pg          = AsyncMock()
        pg.execute.side_effect = [
            _scalars_all(events),
            _scalars_all([]),
            _scalars_all([]),
        ]
        upsert_event_mock = AsyncMock()
        with (
            patch("app.storage.sync.graph_store.upsert_event_node", upsert_event_mock),
            patch("app.storage.sync.graph_store.upsert_entity_node"),
            patch("app.storage.sync.graph_store.upsert_involves_edge"),
            patch("app.storage.sync.graph_store.upsert_causal_edge"),
        ):
            await sync_document(pg, AsyncMock(), doc_id)

        assert upsert_event_mock.await_count == 2
