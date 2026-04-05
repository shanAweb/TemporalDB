"""
tests/unit/test_causal_planner.py

Unit tests for app.query.planners.causal_planner.

embed(), event_store.similarity_search, and graph_store.get_causal_chain
are mocked so no ML model, database, or Neo4j instance is required.

Coverage
--------
  - run() with no seed events → empty PlanResult, confidence 0.0
  - run() seed events found, no chain → plan with seeds, confidence 0.70
  - run() chain records → causal_chain populated, confidence scaled
  - run() confidence capped at 0.90 for large chains
  - run() duplicate chain records deduplicated by event_id
  - run() chain records sorted by hop field
  - run() seed events included in final event list
  - run() extra chain event IDs fetched from PostgreSQL
  - run() document_ids collected from all events
  - run() causal_chain always has 'event_id' key per record
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.query.planners import PlanResult
from app.query.planners import causal_planner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session() -> AsyncMock:
    return AsyncMock()


def _mock_event(doc_id: uuid.UUID | None = None, event_id: uuid.UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=event_id or uuid.uuid4(),
        description="Revenue fell.",
        document_id=doc_id or uuid.uuid4(),
        ts_start=None,
        embedding=[0.1] * 384,
    )


def _chain_record(event_id: str | None = None, hop: int = 1) -> dict:
    return {
        "event_id":    event_id or str(uuid.uuid4()),
        "description": "Inflation caused cost increase.",
        "ts_start":    None,
        "confidence":  0.80,
        "hop":         hop,
    }


def _scalars_all(rows: list) -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.all.return_value = rows
    return r


async def _run(
    *,
    seed_pairs: list[tuple[SimpleNamespace, float]] | None = None,
    chain_records: list[dict] | None = None,
    extra_events: list[SimpleNamespace] | None = None,
    entity_id: uuid.UUID | None = None,
    max_hops: int = 3,
) -> PlanResult:
    """
    Run causal_planner.run() with all external calls mocked.

    seed_pairs      → returned by event_store.similarity_search
    chain_records   → returned by graph_store.get_causal_chain (same for all seeds)
    extra_events    → returned by pg_session.execute for chain event fetch
    entity_id       → passed through to causal_planner.run()
    """
    pg    = _session()
    neo4j = _session()
    seed_pairs    = seed_pairs    or []
    chain_records = chain_records or []
    extra_events  = extra_events  or []

    # pg.execute is called for extra chain event fetch.
    pg.execute.return_value = _scalars_all(extra_events)

    with (
        patch("app.query.planners.causal_planner.embed",
              return_value=[0.1] * 384),
        patch("app.query.planners.causal_planner.event_store.similarity_search",
              return_value=seed_pairs),
        patch("app.query.planners.causal_planner.graph_store.get_causal_chain",
              return_value=chain_records),
    ):
        return await causal_planner.run(
            pg, neo4j, "Why did revenue fall?",
            entity_id=entity_id,
            max_hops=max_hops,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCausalPlannerNoSeeds:
    @pytest.mark.asyncio
    async def test_no_seeds_returns_empty_plan(self) -> None:
        result = await _run(seed_pairs=[])
        assert result.events == []
        assert result.causal_chain == []

    @pytest.mark.asyncio
    async def test_no_seeds_confidence_zero(self) -> None:
        result = await _run(seed_pairs=[])
        assert result.confidence == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_no_seeds_returns_plan_result_type(self) -> None:
        result = await _run(seed_pairs=[])
        assert isinstance(result, PlanResult)


class TestCausalPlannerWithSeeds:
    @pytest.mark.asyncio
    async def test_seed_events_in_result(self) -> None:
        ev = _mock_event()
        result = await _run(seed_pairs=[(ev, 0.2)], chain_records=[])
        assert ev in result.events

    @pytest.mark.asyncio
    async def test_document_ids_from_seed_events(self) -> None:
        doc_id = uuid.uuid4()
        ev = _mock_event(doc_id=doc_id)
        result = await _run(seed_pairs=[(ev, 0.2)], chain_records=[])
        assert doc_id in result.document_ids

    @pytest.mark.asyncio
    async def test_confidence_070_with_empty_chain(self) -> None:
        ev = _mock_event()
        result = await _run(seed_pairs=[(ev, 0.2)], chain_records=[])
        # min(0.90, 0.70 + 0.10 * 0) = 0.70
        assert result.confidence == pytest.approx(0.70)

    @pytest.mark.asyncio
    async def test_confidence_scales_with_chain_length(self) -> None:
        ev = _mock_event()
        chain = [_chain_record() for _ in range(2)]
        result = await _run(seed_pairs=[(ev, 0.2)], chain_records=chain)
        # min(0.90, 0.70 + 0.10 * 2) = 0.90
        assert result.confidence == pytest.approx(0.90)

    @pytest.mark.asyncio
    async def test_confidence_capped_at_090(self) -> None:
        ev = _mock_event()
        # 5 chain records → 0.70 + 0.50 = 1.20, capped at 0.90
        chain = [_chain_record() for _ in range(5)]
        result = await _run(seed_pairs=[(ev, 0.2)], chain_records=chain)
        assert result.confidence <= 0.90


class TestCausalPlannerChain:
    @pytest.mark.asyncio
    async def test_chain_records_populated(self) -> None:
        ev = _mock_event()
        record = _chain_record()
        result = await _run(seed_pairs=[(ev, 0.2)], chain_records=[record])
        assert len(result.causal_chain) == 1
        assert result.causal_chain[0]["event_id"] == record["event_id"]

    @pytest.mark.asyncio
    async def test_duplicate_chain_records_deduplicated(self) -> None:
        ev = _mock_event()
        shared_id = str(uuid.uuid4())
        r1 = _chain_record(event_id=shared_id, hop=1)
        r2 = _chain_record(event_id=shared_id, hop=2)   # same id → deduplicated
        result = await _run(seed_pairs=[(ev, 0.2)], chain_records=[r1, r2])
        matching = [r for r in result.causal_chain if r["event_id"] == shared_id]
        assert len(matching) == 1

    @pytest.mark.asyncio
    async def test_chain_sorted_by_hop(self) -> None:
        ev = _mock_event()
        r_hop2 = _chain_record(hop=2)
        r_hop1 = _chain_record(hop=1)
        result = await _run(seed_pairs=[(ev, 0.2)], chain_records=[r_hop2, r_hop1])
        hops = [r["hop"] for r in result.causal_chain]
        assert hops == sorted(hops)

    @pytest.mark.asyncio
    async def test_extra_chain_events_fetched_from_pg(self) -> None:
        seed_ev  = _mock_event()
        extra_ev = _mock_event()
        record   = _chain_record(event_id=str(extra_ev.id))
        result   = await _run(
            seed_pairs=[(seed_ev, 0.2)],
            chain_records=[record],
            extra_events=[extra_ev],
        )
        event_ids = {ev.id for ev in result.events}
        assert extra_ev.id in event_ids

    @pytest.mark.asyncio
    async def test_seed_events_deduplicated_from_chain_events(self) -> None:
        ev = _mock_event()
        # Chain record references the same event as the seed.
        record = _chain_record(event_id=str(ev.id))
        result = await _run(
            seed_pairs=[(ev, 0.2)],
            chain_records=[record],
            extra_events=[ev],   # pg also returns same event
        )
        # Should appear exactly once in the event list.
        count = sum(1 for e in result.events if e.id == ev.id)
        assert count == 1
