"""
tests/unit/test_temporal_planner.py

Unit tests for app.query.planners.temporal_planner.

event_store.list_events is mocked so no database is required.

Coverage
--------
  - run() delegates to event_store.list_events with correct date bounds
  - run() with time_range → confidence 0.85
  - run() without time_range → confidence 0.60
  - run() with entity_id → forwarded to list_events
  - run() returns empty events when list_events returns nothing
  - document_ids collected from returned events
  - causal_chain always empty for this planner
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.models.schemas.query import TimeRange
from app.query.planners import PlanResult
from app.query.planners import temporal_planner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session() -> AsyncMock:
    return AsyncMock()


def _tr(start_year: int = 2024, end_year: int = 2024) -> TimeRange:
    return TimeRange(
        start=datetime(start_year, 1, 1, tzinfo=timezone.utc),
        end=datetime(end_year, 12, 31, tzinfo=timezone.utc),
    )


def _mock_event(doc_id: uuid.UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        description="Revenue fell.",
        document_id=doc_id or uuid.uuid4(),
        ts_start=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )


async def _run(
    time_range: TimeRange | None = None,
    entity_id: uuid.UUID | None = None,
    events: list | None = None,
    total: int | None = None,
) -> tuple[PlanResult, AsyncMock]:
    session = _session()
    events = events or []
    total  = total  if total is not None else len(events)

    with patch(
        "app.query.planners.temporal_planner.event_store.list_events",
        return_value=(events, total),
    ) as mock_list:
        result = await temporal_planner.run(
            session, time_range, entity_id=entity_id
        )

    return result, mock_list


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTemporalPlannerConfidence:
    @pytest.mark.asyncio
    async def test_with_time_range_confidence_085(self) -> None:
        result, _ = await _run(time_range=_tr())
        assert result.confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_without_time_range_confidence_060(self) -> None:
        result, _ = await _run(time_range=None)
        assert result.confidence == pytest.approx(0.60)


class TestTemporalPlannerDelegation:
    @pytest.mark.asyncio
    async def test_from_date_passed_when_time_range_set(self) -> None:
        tr = _tr(2024, 2024)
        _, mock_list = await _run(time_range=tr)
        kwargs = mock_list.call_args.kwargs
        assert kwargs["from_date"] == tr.start

    @pytest.mark.asyncio
    async def test_to_date_passed_when_time_range_set(self) -> None:
        tr = _tr(2024, 2024)
        _, mock_list = await _run(time_range=tr)
        kwargs = mock_list.call_args.kwargs
        assert kwargs["to_date"] == tr.end

    @pytest.mark.asyncio
    async def test_from_to_date_none_without_time_range(self) -> None:
        _, mock_list = await _run(time_range=None)
        kwargs = mock_list.call_args.kwargs
        assert kwargs["from_date"] is None
        assert kwargs["to_date"]   is None

    @pytest.mark.asyncio
    async def test_entity_id_forwarded(self) -> None:
        eid = uuid.uuid4()
        _, mock_list = await _run(entity_id=eid)
        assert mock_list.call_args.kwargs["entity_id"] == eid

    @pytest.mark.asyncio
    async def test_entity_id_none_when_not_supplied(self) -> None:
        _, mock_list = await _run()
        assert mock_list.call_args.kwargs["entity_id"] is None


class TestTemporalPlannerResult:
    @pytest.mark.asyncio
    async def test_events_populated(self) -> None:
        ev = _mock_event()
        result, _ = await _run(events=[ev])
        assert result.events == [ev]

    @pytest.mark.asyncio
    async def test_empty_events_on_no_results(self) -> None:
        result, _ = await _run(events=[])
        assert result.events == []

    @pytest.mark.asyncio
    async def test_document_ids_collected(self) -> None:
        doc_id = uuid.uuid4()
        ev = _mock_event(doc_id=doc_id)
        result, _ = await _run(events=[ev])
        assert doc_id in result.document_ids

    @pytest.mark.asyncio
    async def test_causal_chain_always_empty(self) -> None:
        ev = _mock_event()
        result, _ = await _run(events=[ev])
        assert result.causal_chain == []

    @pytest.mark.asyncio
    async def test_returns_plan_result_type(self) -> None:
        result, _ = await _run()
        assert isinstance(result, PlanResult)
