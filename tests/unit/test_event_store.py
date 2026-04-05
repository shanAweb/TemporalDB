"""
tests/unit/test_event_store.py

Unit tests for app.storage.event_store.

All tests use an AsyncMock SQLAlchemy session — no real database is
required.  The mock session's execute() return value is configured per
test to simulate the ORM query results.

Coverage
--------
  - insert_event: session.add + flush + refresh called; event returned
  - get_event_by_id: found → event; not found → None
  - list_events: (results, total) tuple returned; count query + data query
  - list_events: empty result set returns ([], 0)
  - delete_event: found → True; not found → False
  - link_entities_to_event: empty list is no-op; new links added; duplicates skipped
  - insert_causal_relation: session.add + flush + refresh called; relation returned
  - get_causal_relations: both directions; as_cause=False; as_effect=False → []
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.storage.event_store import (
    delete_causal_relation,
    delete_event,
    get_causal_relations,
    get_event_by_id,
    insert_causal_relation,
    insert_event,
    link_entities_to_event,
    list_events,
)


# ---------------------------------------------------------------------------
# Session mock helpers
# ---------------------------------------------------------------------------

def _session() -> AsyncMock:
    """Return a bare AsyncSession mock."""
    return AsyncMock()


def _scalar_one_result(value: object) -> MagicMock:
    result = MagicMock()
    result.scalar_one.return_value = value
    return result


def _scalar_one_or_none_result(value: object) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _scalars_all_result(rows: list) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    return result


def _mock_event(description: str = "Costs rose.") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        description=description,
        event_type="state_change",
        document_id=uuid.uuid4(),
        confidence=0.85,
    )


def _mock_relation() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        cause_event_id=uuid.uuid4(),
        effect_event_id=uuid.uuid4(),
        confidence=0.90,
        evidence=None,
    )


# ---------------------------------------------------------------------------
# insert_event
# ---------------------------------------------------------------------------

class TestInsertEvent:
    @pytest.mark.asyncio
    async def test_session_add_called(self) -> None:
        session = _session()
        await insert_event(session, description="Revenue fell.", document_id=uuid.uuid4(), confidence=0.9)
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_called(self) -> None:
        session = _session()
        await insert_event(session, description="Revenue fell.", document_id=uuid.uuid4(), confidence=0.9)
        session.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_refresh_called(self) -> None:
        session = _session()
        await insert_event(session, description="Revenue fell.", document_id=uuid.uuid4(), confidence=0.9)
        session.refresh.assert_awaited()

    @pytest.mark.asyncio
    async def test_returns_event_object(self) -> None:
        session = _session()
        result = await insert_event(session, description="Revenue fell.", document_id=uuid.uuid4(), confidence=0.9)
        # Should return the Event instance (not None, not the session mock).
        assert result is not None


# ---------------------------------------------------------------------------
# get_event_by_id
# ---------------------------------------------------------------------------

class TestGetEventById:
    @pytest.mark.asyncio
    async def test_found_returns_event(self) -> None:
        ev = _mock_event()
        session = _session()
        session.execute.return_value = _scalar_one_or_none_result(ev)
        result = await get_event_by_id(session, ev.id)
        assert result is ev

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none_result(None)
        result = await get_event_by_id(session, uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_called_once(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none_result(None)
        await get_event_by_id(session, uuid.uuid4())
        session.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# list_events
# ---------------------------------------------------------------------------

class TestListEvents:
    @pytest.mark.asyncio
    async def test_returns_events_and_total(self) -> None:
        ev = _mock_event()
        session = _session()
        # list_events calls execute() twice: count query then data query.
        session.execute.side_effect = [
            _scalar_one_result(1),       # count
            _scalars_all_result([ev]),    # data
        ]
        events, total = await list_events(session)
        assert total == 1
        assert events == [ev]

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        session = _session()
        session.execute.side_effect = [
            _scalar_one_result(0),
            _scalars_all_result([]),
        ]
        events, total = await list_events(session)
        assert events == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_execute_called_twice(self) -> None:
        session = _session()
        session.execute.side_effect = [
            _scalar_one_result(0),
            _scalars_all_result([]),
        ]
        await list_events(session)
        assert session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_multiple_events_returned(self) -> None:
        events = [_mock_event(f"Event {i}") for i in range(5)]
        session = _session()
        session.execute.side_effect = [
            _scalar_one_result(5),
            _scalars_all_result(events),
        ]
        result_events, total = await list_events(session)
        assert total == 5
        assert len(result_events) == 5


# ---------------------------------------------------------------------------
# delete_event
# ---------------------------------------------------------------------------

class TestDeleteEvent:
    @pytest.mark.asyncio
    async def test_found_returns_true(self) -> None:
        event_id = uuid.uuid4()
        session = _session()
        session.execute.return_value = _scalar_one_or_none_result(event_id)
        result = await delete_event(session, event_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_not_found_returns_false(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none_result(None)
        result = await delete_event(session, uuid.uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_flush_called_after_delete(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none_result(None)
        await delete_event(session, uuid.uuid4())
        session.flush.assert_awaited()


# ---------------------------------------------------------------------------
# link_entities_to_event
# ---------------------------------------------------------------------------

class TestLinkEntitiesToEvent:
    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self) -> None:
        session = _session()
        await link_entities_to_event(session, uuid.uuid4(), [])
        session.execute.assert_not_awaited()
        session.add_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_links_added(self) -> None:
        event_id = uuid.uuid4()
        entity_id = uuid.uuid4()
        session = _session()
        # No existing links.
        session.execute.return_value = _scalars_all_result([])
        await link_entities_to_event(session, event_id, [entity_id])
        session.add_all.assert_called_once()
        added = session.add_all.call_args[0][0]
        assert len(added) == 1

    @pytest.mark.asyncio
    async def test_already_linked_entity_skipped(self) -> None:
        event_id = uuid.uuid4()
        entity_id = uuid.uuid4()
        session = _session()
        # entity_id already exists in the existing links.
        session.execute.return_value = _scalars_all_result([entity_id])
        await link_entities_to_event(session, event_id, [entity_id])
        # add_all should not be called since all entities are already linked.
        session.add_all.assert_not_called()


# ---------------------------------------------------------------------------
# insert_causal_relation
# ---------------------------------------------------------------------------

class TestInsertCausalRelation:
    @pytest.mark.asyncio
    async def test_session_add_called(self) -> None:
        session = _session()
        await insert_causal_relation(
            session,
            cause_event_id=uuid.uuid4(),
            effect_event_id=uuid.uuid4(),
            confidence=0.85,
        )
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_and_refresh_called(self) -> None:
        session = _session()
        await insert_causal_relation(
            session,
            cause_event_id=uuid.uuid4(),
            effect_event_id=uuid.uuid4(),
        )
        session.flush.assert_awaited()
        session.refresh.assert_awaited()

    @pytest.mark.asyncio
    async def test_returns_relation_object(self) -> None:
        session = _session()
        result = await insert_causal_relation(
            session,
            cause_event_id=uuid.uuid4(),
            effect_event_id=uuid.uuid4(),
            confidence=0.90,
            evidence="Costs rose because of inflation.",
        )
        assert result is not None


# ---------------------------------------------------------------------------
# get_causal_relations
# ---------------------------------------------------------------------------

class TestGetCausalRelations:
    @pytest.mark.asyncio
    async def test_both_false_returns_empty_without_query(self) -> None:
        session = _session()
        result = await get_causal_relations(
            session, uuid.uuid4(), as_cause=False, as_effect=False
        )
        assert result == []
        session.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_as_cause_true_returns_relations(self) -> None:
        relation = _mock_relation()
        session = _session()
        session.execute.return_value = _scalars_all_result([relation])
        result = await get_causal_relations(
            session, uuid.uuid4(), as_cause=True, as_effect=False
        )
        assert result == [relation]

    @pytest.mark.asyncio
    async def test_both_true_queries_both_directions(self) -> None:
        relation = _mock_relation()
        session = _session()
        session.execute.return_value = _scalars_all_result([relation])
        result = await get_causal_relations(
            session, uuid.uuid4(), as_cause=True, as_effect=True
        )
        assert len(result) == 1
        session.execute.assert_awaited_once()
