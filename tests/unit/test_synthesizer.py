"""
tests/unit/test_synthesizer.py

Unit tests for app.query.synthesizer.

Tests the pure helper functions (_events_to_brief, _chain_to_links,
_fallback_answer, _format_events, _format_chain, _format_sources) and
the main synthesize() function with a mocked Ollama client and mocked
PostgreSQL session so no running infrastructure is required.

Coverage
--------
  - _events_to_brief: ORM event objects → EventBrief list
  - _chain_to_links: chain dicts → CausalChainLink list; bad records skipped
  - _fallback_answer: causal chain path, events-only path, empty path
  - _format_events / _format_chain / _format_sources: string formatting
  - synthesize(): LLM answer returned on success
  - synthesize(): fallback answer used when Ollama raises
  - synthesize(): confidence rounded from plan
  - synthesize(): intent value propagated to response
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas.event import EventBrief
from app.models.schemas.query import CausalChainLink, SourceReference
from app.query.intent import Intent
from app.query.planners import PlanResult
from app.query.synthesizer import (
    _chain_to_links,
    _events_to_brief,
    _fallback_answer,
    _format_chain,
    _format_events,
    _format_sources,
    synthesize,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _orm_event(
    description: str = "Costs rose.",
    ts_start: datetime | None = None,
    confidence: float = 0.85,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        description=description,
        ts_start=ts_start,
        confidence=confidence,
    )


def _chain_record(
    event_id: str | None = None,
    description: str = "Inflation led to cost increases.",
    ts_start: str | None = "2024-07-01T00:00:00",
    confidence: float = 0.80,
) -> dict:
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "description": description,
        "ts_start": ts_start,
        "confidence": confidence,
    }


def _brief(description: str = "Costs rose.", ts_start: datetime | None = None) -> EventBrief:
    return EventBrief(
        id=uuid.uuid4(),
        description=description,
        ts_start=ts_start,
        confidence=0.85,
    )


def _link(description: str = "Inflation caused delays.") -> CausalChainLink:
    return CausalChainLink(
        id=uuid.uuid4(),
        description=description,
        ts_start=datetime(2024, 7, 1, tzinfo=timezone.utc),
        confidence=0.80,
    )


def _empty_plan(confidence: float = 0.75) -> PlanResult:
    return PlanResult(events=[], causal_chain=[], document_ids=set(), confidence=confidence)


def _mock_pg_session(docs: list = []) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = docs
    session.execute.return_value = result
    return session


# ---------------------------------------------------------------------------
# _events_to_brief
# ---------------------------------------------------------------------------

class TestEventsToBrief:
    def test_empty_plan_returns_empty(self) -> None:
        plan = _empty_plan()
        assert _events_to_brief(plan) == []

    def test_converts_orm_event(self) -> None:
        ts = datetime(2024, 7, 1, tzinfo=timezone.utc)
        ev = _orm_event("Revenue fell.", ts_start=ts, confidence=0.90)
        plan = PlanResult(events=[ev], causal_chain=[], document_ids=set(), confidence=0.8)
        briefs = _events_to_brief(plan)
        assert len(briefs) == 1
        assert briefs[0].description == "Revenue fell."
        assert briefs[0].ts_start == ts
        assert briefs[0].confidence == pytest.approx(0.90)

    def test_multiple_events_preserved(self) -> None:
        events = [_orm_event(f"Event {i}") for i in range(3)]
        plan = PlanResult(events=events, causal_chain=[], document_ids=set(), confidence=0.7)
        briefs = _events_to_brief(plan)
        assert len(briefs) == 3

    def test_none_ts_start_preserved(self) -> None:
        ev = _orm_event(ts_start=None)
        plan = PlanResult(events=[ev], causal_chain=[], document_ids=set(), confidence=0.7)
        briefs = _events_to_brief(plan)
        assert briefs[0].ts_start is None


# ---------------------------------------------------------------------------
# _chain_to_links
# ---------------------------------------------------------------------------

class TestChainToLinks:
    def test_empty_chain_returns_empty(self) -> None:
        plan = _empty_plan()
        assert _chain_to_links(plan) == []

    def test_valid_record_converts(self) -> None:
        record = _chain_record()
        plan = PlanResult(events=[], causal_chain=[record], document_ids=set(), confidence=0.8)
        links = _chain_to_links(plan)
        assert len(links) == 1
        assert links[0].description == record["description"]
        assert links[0].confidence == pytest.approx(0.80)

    def test_ts_start_parsed_from_iso(self) -> None:
        record = _chain_record(ts_start="2024-07-01T00:00:00")
        plan = PlanResult(events=[], causal_chain=[record], document_ids=set(), confidence=0.8)
        links = _chain_to_links(plan)
        assert links[0].ts_start is not None
        assert links[0].ts_start.year == 2024

    def test_none_ts_start_preserved(self) -> None:
        record = _chain_record(ts_start=None)
        plan = PlanResult(events=[], causal_chain=[record], document_ids=set(), confidence=0.8)
        links = _chain_to_links(plan)
        assert links[0].ts_start is None

    def test_invalid_event_id_skipped(self) -> None:
        bad = _chain_record(event_id="not-a-uuid")
        plan = PlanResult(events=[], causal_chain=[bad], document_ids=set(), confidence=0.8)
        links = _chain_to_links(plan)
        assert links == []

    def test_missing_event_id_skipped(self) -> None:
        record = {"description": "Missing id.", "confidence": 0.7}
        plan = PlanResult(events=[], causal_chain=[record], document_ids=set(), confidence=0.8)
        links = _chain_to_links(plan)
        assert links == []

    def test_multiple_records(self) -> None:
        records = [_chain_record(), _chain_record()]
        plan = PlanResult(events=[], causal_chain=records, document_ids=set(), confidence=0.8)
        assert len(_chain_to_links(plan)) == 2


# ---------------------------------------------------------------------------
# _fallback_answer
# ---------------------------------------------------------------------------

class TestFallbackAnswer:
    def test_no_events_no_chain(self) -> None:
        result = _fallback_answer("Why did costs rise?", [], [])
        assert "No relevant events" in result

    def test_causal_chain_path(self) -> None:
        links = [_link("Inflation caused cost increases.")]
        result = _fallback_answer("Why did costs rise?", [], links)
        assert "Causal chain" in result
        assert "Inflation caused cost increases." in result

    def test_events_only_path(self) -> None:
        briefs = [_brief("Revenue fell.")]
        result = _fallback_answer("What happened?", briefs, [])
        assert "Relevant events" in result
        assert "Revenue fell." in result

    def test_chain_wins_over_events(self) -> None:
        briefs = [_brief("Revenue fell.")]
        links = [_link("Inflation caused delays.")]
        result = _fallback_answer("Why?", briefs, links)
        # Causal chain takes priority.
        assert "Causal chain" in result

    def test_question_included_in_output(self) -> None:
        result = _fallback_answer("Why did costs rise?", [], [])
        assert "Why did costs rise?" in result


# ---------------------------------------------------------------------------
# _format_events / _format_chain / _format_sources
# ---------------------------------------------------------------------------

class TestFormatHelpers:
    def test_format_events_empty(self) -> None:
        assert "No events" in _format_events([])

    def test_format_events_with_items(self) -> None:
        ts = datetime(2024, 7, 1, tzinfo=timezone.utc)
        brief = _brief("Revenue fell.", ts_start=ts)
        result = _format_events([brief])
        assert "Revenue fell." in result
        assert "2024" in result

    def test_format_chain_empty(self) -> None:
        assert "No causal chain" in _format_chain([])

    def test_format_chain_with_items(self) -> None:
        link = _link("Inflation caused cost increases.")
        result = _format_chain([link])
        assert "Inflation caused cost increases." in result
        assert "1." in result

    def test_format_sources_empty(self) -> None:
        assert "No source" in _format_sources([])

    def test_format_sources_with_items(self) -> None:
        src = SourceReference(id=uuid.uuid4(), source="quarterly-report", metadata=None)
        result = _format_sources([src])
        assert "quarterly-report" in result


# ---------------------------------------------------------------------------
# synthesize()
# ---------------------------------------------------------------------------

class TestSynthesize:
    @pytest.mark.asyncio
    async def test_llm_answer_returned_on_success(self) -> None:
        plan = _empty_plan(confidence=0.80)
        session = _mock_pg_session()
        mock_generate = AsyncMock(return_value="The answer is here.")

        with patch("app.query.synthesizer.ollama_client") as mock_client:
            mock_client.generate = mock_generate
            response = await synthesize(session, plan, "Why did revenue drop?", Intent.CAUSAL_WHY)

        assert response.answer == "The answer is here."
        mock_generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fallback_used_when_llm_raises(self) -> None:
        plan = _empty_plan(confidence=0.70)
        session = _mock_pg_session()
        mock_generate = AsyncMock(side_effect=RuntimeError("Ollama unavailable"))

        with patch("app.query.synthesizer.ollama_client") as mock_client:
            mock_client.generate = mock_generate
            response = await synthesize(session, plan, "Why did costs rise?", Intent.CAUSAL_WHY)

        # Must not raise; answer comes from _fallback_answer.
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

    @pytest.mark.asyncio
    async def test_confidence_rounded_from_plan(self) -> None:
        plan = _empty_plan(confidence=0.7777)
        session = _mock_pg_session()

        with patch("app.query.synthesizer.ollama_client") as mock_client:
            mock_client.generate = AsyncMock(return_value="Answer.")
            response = await synthesize(session, plan, "Why?", Intent.SIMILARITY)

        assert response.confidence == pytest.approx(0.7777, abs=1e-4)

    @pytest.mark.asyncio
    async def test_intent_value_in_response(self) -> None:
        plan = _empty_plan()
        session = _mock_pg_session()

        with patch("app.query.synthesizer.ollama_client") as mock_client:
            mock_client.generate = AsyncMock(return_value="Answer.")
            response = await synthesize(session, plan, "What happened?", Intent.TEMPORAL_RANGE)

        assert response.intent == Intent.TEMPORAL_RANGE.value

    @pytest.mark.asyncio
    async def test_events_and_chain_populated_in_response(self) -> None:
        ev = _orm_event("Costs rose.")
        record = _chain_record()
        plan = PlanResult(events=[ev], causal_chain=[record], document_ids=set(), confidence=0.8)
        session = _mock_pg_session()

        with patch("app.query.synthesizer.ollama_client") as mock_client:
            mock_client.generate = AsyncMock(return_value="Answer.")
            response = await synthesize(session, plan, "Why?", Intent.CAUSAL_WHY)

        assert len(response.events) == 1
        assert len(response.causal_chain) == 1
