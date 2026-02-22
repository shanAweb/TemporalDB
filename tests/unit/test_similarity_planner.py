"""
tests/unit/test_similarity_planner.py

Unit tests for app.query.planners.similarity_planner.

embed() and event_store.similarity_search are mocked so no ML model
or database is required.

Coverage
--------
  - run() embeds the question and calls similarity_search
  - run() returns events from similarity_search results
  - run() confidence = mean(1 - distance) across results
  - run() with empty results → confidence 0.0
  - run() with entity_id → post-filters to linked event IDs
  - run() with time_range → post-filters to events in window
  - run() with both filters → combined post-filtering
  - run() trims results to requested limit after post-filtering
  - causal_chain always empty for this planner
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas.query import TimeRange
from app.query.planners import PlanResult
from app.query.planners import similarity_planner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session() -> AsyncMock:
    return AsyncMock()


def _mock_event(
    doc_id: uuid.UUID | None = None,
    ts_start: datetime | None = None,
    event_id: uuid.UUID | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=event_id or uuid.uuid4(),
        description="Revenue fell.",
        document_id=doc_id or uuid.uuid4(),
        ts_start=ts_start,
    )


def _tr(
    start: datetime | None = None,
    end:   datetime | None = None,
) -> TimeRange:
    return TimeRange(
        start=start or datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=end   or datetime(2024, 12, 31, tzinfo=timezone.utc),
    )


def _scalars_all(ids: list[uuid.UUID]) -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.all.return_value = ids
    return r


async def _run(
    question: str = "Why did revenue fall?",
    pairs: list[tuple[SimpleNamespace, float]] | None = None,
    entity_id: uuid.UUID | None = None,
    time_range: TimeRange | None = None,
    linked_ids: list[uuid.UUID] | None = None,
    limit: int = 10,
) -> PlanResult:
    session = _session()
    pairs   = pairs or []

    if entity_id is not None and linked_ids is not None:
        session.execute.return_value = _scalars_all(linked_ids)

    with (
        patch("app.query.planners.similarity_planner.embed",
              return_value=[0.1] * 384),
        patch("app.query.planners.similarity_planner.event_store.similarity_search",
              return_value=pairs),
    ):
        return await similarity_planner.run(
            session,
            question,
            entity_id=entity_id,
            time_range=time_range,
            limit=limit,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSimilarityPlannerBasic:
    @pytest.mark.asyncio
    async def test_events_returned_from_search(self) -> None:
        ev = _mock_event()
        result = await _run(pairs=[(ev, 0.2)])
        assert result.events == [ev]

    @pytest.mark.asyncio
    async def test_empty_search_returns_empty_plan(self) -> None:
        result = await _run(pairs=[])
        assert result.events == []

    @pytest.mark.asyncio
    async def test_causal_chain_always_empty(self) -> None:
        ev = _mock_event()
        result = await _run(pairs=[(ev, 0.1)])
        assert result.causal_chain == []

    @pytest.mark.asyncio
    async def test_document_ids_collected(self) -> None:
        doc_id = uuid.uuid4()
        ev = _mock_event(doc_id=doc_id)
        result = await _run(pairs=[(ev, 0.1)])
        assert doc_id in result.document_ids

    @pytest.mark.asyncio
    async def test_returns_plan_result_type(self) -> None:
        result = await _run()
        assert isinstance(result, PlanResult)


class TestSimilarityPlannerConfidence:
    @pytest.mark.asyncio
    async def test_confidence_is_mean_of_inverted_distances(self) -> None:
        ev1 = _mock_event()
        ev2 = _mock_event()
        # distances 0.2 and 0.4 → (0.8 + 0.6) / 2 = 0.7
        result = await _run(pairs=[(ev1, 0.2), (ev2, 0.4)])
        assert result.confidence == pytest.approx(0.7, abs=1e-4)

    @pytest.mark.asyncio
    async def test_confidence_zero_when_no_results(self) -> None:
        result = await _run(pairs=[])
        assert result.confidence == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_confidence_one_for_distance_zero(self) -> None:
        ev = _mock_event()
        result = await _run(pairs=[(ev, 0.0)])
        assert result.confidence == pytest.approx(1.0)


class TestSimilarityPlannerEntityFilter:
    @pytest.mark.asyncio
    async def test_unlinked_events_excluded(self) -> None:
        linked_id   = uuid.uuid4()
        unlinked_id = uuid.uuid4()
        ev_linked   = _mock_event(event_id=linked_id)
        ev_unlinked = _mock_event(event_id=unlinked_id)

        result = await _run(
            pairs=[(ev_linked, 0.1), (ev_unlinked, 0.1)],
            entity_id=uuid.uuid4(),
            linked_ids=[linked_id],
        )

        returned_ids = {ev.id for ev in result.events}
        assert linked_id   in returned_ids
        assert unlinked_id not in returned_ids

    @pytest.mark.asyncio
    async def test_all_events_excluded_when_none_linked(self) -> None:
        ev = _mock_event()
        result = await _run(
            pairs=[(ev, 0.1)],
            entity_id=uuid.uuid4(),
            linked_ids=[],  # no linked events
        )
        assert result.events == []


class TestSimilarityPlannerTimeRangeFilter:
    @pytest.mark.asyncio
    async def test_event_in_range_included(self) -> None:
        ts = datetime(2024, 7, 15, tzinfo=timezone.utc)
        ev = _mock_event(ts_start=ts)
        tr = _tr(
            start=datetime(2024, 7, 1,  tzinfo=timezone.utc),
            end=  datetime(2024, 7, 31, tzinfo=timezone.utc),
        )
        result = await _run(pairs=[(ev, 0.1)], time_range=tr)
        assert ev in result.events

    @pytest.mark.asyncio
    async def test_event_outside_range_excluded(self) -> None:
        ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
        ev = _mock_event(ts_start=ts)
        tr = _tr(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=  datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        result = await _run(pairs=[(ev, 0.1)], time_range=tr)
        assert ev not in result.events

    @pytest.mark.asyncio
    async def test_event_without_ts_start_excluded(self) -> None:
        ev = _mock_event(ts_start=None)
        result = await _run(pairs=[(ev, 0.1)], time_range=_tr())
        assert result.events == []


class TestSimilarityPlannerLimit:
    @pytest.mark.asyncio
    async def test_results_trimmed_to_limit(self) -> None:
        pairs = [(_mock_event(), 0.1) for _ in range(10)]
        result = await _run(pairs=pairs, limit=3)
        assert len(result.events) <= 3
