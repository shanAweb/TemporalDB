"""
tests/integration/test_query_orchestrator.py

Integration tests for app.query.orchestrator.handle_query().

All external boundaries (intent classifier, time-range extractor, entity
resolver, planners, synthesizer) are mocked so no running infrastructure
is required.

Coverage
--------
  - CAUSAL_WHY intent → causal_planner.run() called
  - TEMPORAL_RANGE intent → temporal_planner.run() called
  - SIMILARITY intent → similarity_planner.run() called
  - ENTITY_TIMELINE intent → entity_planner.run() called
  - Entity filter resolved and forwarded to planner
  - Time range extracted and forwarded to planner
  - Synthesizer always called with the planner's PlanResult
  - Final QueryResponse fields propagated correctly
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas.query import (
    CausalChainLink,
    QueryRequest,
    QueryResponse,
    TimeRange,
)
from app.query.intent import Intent, IntentResult
from app.query.orchestrator import handle_query
from app.query.planners import PlanResult


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _intent_result(intent: Intent) -> IntentResult:
    return IntentResult(intent=intent, confidence=0.90, method="heuristic")


def _empty_plan(confidence: float = 0.75) -> PlanResult:
    return PlanResult(events=[], causal_chain=[], document_ids=set(), confidence=confidence)


def _query_response(intent: Intent) -> QueryResponse:
    return QueryResponse(
        answer="Synthesized answer.",
        confidence=0.80,
        intent=intent.value,
        causal_chain=[],
        events=[],
        sources=[],
    )


def _make_sessions() -> tuple[AsyncMock, AsyncMock]:
    """Return (pg_session, neo4j_session) mocks."""
    return AsyncMock(), AsyncMock()


# ---------------------------------------------------------------------------
# Planner dispatch tests
# ---------------------------------------------------------------------------

class TestPlannerDispatch:
    """handle_query must route each intent to the correct planner."""

    async def _run(
        self,
        intent: Intent,
        question: str = "test question",
        entity_filter: str | None = None,
        time_range: TimeRange | None = None,
    ) -> tuple[QueryResponse, MagicMock, MagicMock, MagicMock, MagicMock]:
        """
        Run handle_query with all boundaries mocked.

        Returns (response, causal_mock, temporal_mock, similarity_mock, entity_mock).
        """
        pg, neo4j = _make_sessions()
        request = QueryRequest(
            question=question,
            entity_filter=entity_filter,
            time_range=time_range,
        )
        plan = _empty_plan()
        response = _query_response(intent)

        causal_run    = AsyncMock(return_value=plan)
        temporal_run  = AsyncMock(return_value=plan)
        similarity_run = AsyncMock(return_value=plan)
        entity_run    = AsyncMock(return_value=plan)

        with (
            patch("app.query.orchestrator.classify_intent",
                  return_value=_intent_result(intent)),
            patch("app.query.orchestrator.extract_time_range", return_value=None),
            patch("app.query.orchestrator.resolve_entity_filter", return_value=None),
            patch("app.query.orchestrator.causal_planner.run",    causal_run),
            patch("app.query.orchestrator.temporal_planner.run",  temporal_run),
            patch("app.query.orchestrator.similarity_planner.run", similarity_run),
            patch("app.query.orchestrator.entity_planner.run",    entity_run),
            patch("app.query.orchestrator.synthesizer.synthesize",
                  return_value=response),
        ):
            resp = await handle_query(request, pg, neo4j)

        return resp, causal_run, temporal_run, similarity_run, entity_run

    @pytest.mark.asyncio
    async def test_causal_why_calls_causal_planner(self) -> None:
        _, causal, temporal, similarity, entity = await self._run(Intent.CAUSAL_WHY)
        causal.assert_awaited_once()
        temporal.assert_not_awaited()
        similarity.assert_not_awaited()
        entity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_temporal_range_calls_temporal_planner(self) -> None:
        _, causal, temporal, similarity, entity = await self._run(Intent.TEMPORAL_RANGE)
        temporal.assert_awaited_once()
        causal.assert_not_awaited()
        similarity.assert_not_awaited()
        entity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_similarity_calls_similarity_planner(self) -> None:
        _, causal, temporal, similarity, entity = await self._run(Intent.SIMILARITY)
        similarity.assert_awaited_once()
        causal.assert_not_awaited()
        temporal.assert_not_awaited()
        entity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_entity_timeline_calls_entity_planner(self) -> None:
        _, causal, temporal, similarity, entity = await self._run(Intent.ENTITY_TIMELINE)
        entity.assert_awaited_once()
        causal.assert_not_awaited()
        temporal.assert_not_awaited()
        similarity.assert_not_awaited()


# ---------------------------------------------------------------------------
# Filter forwarding tests
# ---------------------------------------------------------------------------

class TestFilterForwarding:
    """Resolved entity_id and time_range must be forwarded to the planner."""

    @pytest.mark.asyncio
    async def test_entity_id_forwarded_to_causal_planner(self) -> None:
        pg, neo4j = _make_sessions()
        entity_id = uuid.uuid4()
        request = QueryRequest(question="Why did costs rise?", entity_filter="Acme Corp")
        plan = _empty_plan()
        causal_run = AsyncMock(return_value=plan)

        with (
            patch("app.query.orchestrator.classify_intent",
                  return_value=_intent_result(Intent.CAUSAL_WHY)),
            patch("app.query.orchestrator.extract_time_range", return_value=None),
            patch("app.query.orchestrator.resolve_entity_filter",
                  return_value=entity_id),
            patch("app.query.orchestrator.causal_planner.run", causal_run),
            patch("app.query.orchestrator.synthesizer.synthesize",
                  return_value=_query_response(Intent.CAUSAL_WHY)),
        ):
            await handle_query(request, pg, neo4j)

        call_kwargs = causal_run.call_args
        assert call_kwargs.kwargs.get("entity_id") == entity_id

    @pytest.mark.asyncio
    async def test_time_range_forwarded_to_temporal_planner(self) -> None:
        pg, neo4j = _make_sessions()
        tr = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 31, tzinfo=timezone.utc),
        )
        request = QueryRequest(question="What happened in Q1 2024?")
        plan = _empty_plan()
        temporal_run = AsyncMock(return_value=plan)

        with (
            patch("app.query.orchestrator.classify_intent",
                  return_value=_intent_result(Intent.TEMPORAL_RANGE)),
            patch("app.query.orchestrator.extract_time_range", return_value=tr),
            patch("app.query.orchestrator.resolve_entity_filter", return_value=None),
            patch("app.query.orchestrator.temporal_planner.run", temporal_run),
            patch("app.query.orchestrator.synthesizer.synthesize",
                  return_value=_query_response(Intent.TEMPORAL_RANGE)),
        ):
            await handle_query(request, pg, neo4j)

        positional_args = temporal_run.call_args.args
        # Signature: run(pg_session, time_range, *, entity_id)
        assert positional_args[1] == tr

    @pytest.mark.asyncio
    async def test_no_entity_filter_passes_none(self) -> None:
        pg, neo4j = _make_sessions()
        request = QueryRequest(question="Find events similar to supply chain issues.")
        plan = _empty_plan()
        similarity_run = AsyncMock(return_value=plan)

        with (
            patch("app.query.orchestrator.classify_intent",
                  return_value=_intent_result(Intent.SIMILARITY)),
            patch("app.query.orchestrator.extract_time_range", return_value=None),
            patch("app.query.orchestrator.resolve_entity_filter", return_value=None),
            patch("app.query.orchestrator.similarity_planner.run", similarity_run),
            patch("app.query.orchestrator.synthesizer.synthesize",
                  return_value=_query_response(Intent.SIMILARITY)),
        ):
            await handle_query(request, pg, neo4j)

        assert similarity_run.call_args.kwargs.get("entity_id") is None


# ---------------------------------------------------------------------------
# Synthesizer integration
# ---------------------------------------------------------------------------

class TestSynthesizerIntegration:
    """Synthesizer must always be called and its response returned verbatim."""

    @pytest.mark.asyncio
    async def test_synthesizer_called_with_plan(self) -> None:
        pg, neo4j = _make_sessions()
        request = QueryRequest(question="Why did revenue drop?")
        plan = _empty_plan(confidence=0.88)
        expected_response = _query_response(Intent.CAUSAL_WHY)
        synth_mock = AsyncMock(return_value=expected_response)

        with (
            patch("app.query.orchestrator.classify_intent",
                  return_value=_intent_result(Intent.CAUSAL_WHY)),
            patch("app.query.orchestrator.extract_time_range", return_value=None),
            patch("app.query.orchestrator.resolve_entity_filter", return_value=None),
            patch("app.query.orchestrator.causal_planner.run",
                  AsyncMock(return_value=plan)),
            patch("app.query.orchestrator.synthesizer.synthesize", synth_mock),
        ):
            response = await handle_query(request, pg, neo4j)

        synth_mock.assert_awaited_once()
        # The plan passed to synthesize must be the one returned by the planner.
        synth_call_args = synth_mock.call_args
        assert synth_call_args.args[1] is plan
        assert response is expected_response

    @pytest.mark.asyncio
    async def test_response_fields_propagated(self) -> None:
        pg, neo4j = _make_sessions()
        request = QueryRequest(question="What happened in Q3 2024?")
        plan = _empty_plan()
        expected = QueryResponse(
            answer="Supply chain delays peaked in Q3.",
            confidence=0.92,
            intent=Intent.TEMPORAL_RANGE.value,
            causal_chain=[
                CausalChainLink(
                    id=uuid.uuid4(),
                    description="Inflation caused delay.",
                    ts_start=datetime(2024, 7, 1, tzinfo=timezone.utc),
                    confidence=0.85,
                )
            ],
            events=[],
            sources=[],
        )

        with (
            patch("app.query.orchestrator.classify_intent",
                  return_value=_intent_result(Intent.TEMPORAL_RANGE)),
            patch("app.query.orchestrator.extract_time_range", return_value=None),
            patch("app.query.orchestrator.resolve_entity_filter", return_value=None),
            patch("app.query.orchestrator.temporal_planner.run",
                  AsyncMock(return_value=plan)),
            patch("app.query.orchestrator.synthesizer.synthesize",
                  AsyncMock(return_value=expected)),
        ):
            response = await handle_query(request, pg, neo4j)

        assert response.answer == "Supply chain delays peaked in Q3."
        assert response.confidence == pytest.approx(0.92)
        assert response.intent == Intent.TEMPORAL_RANGE.value
        assert len(response.causal_chain) == 1


# ---------------------------------------------------------------------------
# max_causal_hops forwarding
# ---------------------------------------------------------------------------

class TestCausalHopsForwarding:
    @pytest.mark.asyncio
    async def test_max_hops_forwarded_to_causal_planner(self) -> None:
        pg, neo4j = _make_sessions()
        request = QueryRequest(question="Why did this happen?", max_causal_hops=5)
        plan = _empty_plan()
        causal_run = AsyncMock(return_value=plan)

        with (
            patch("app.query.orchestrator.classify_intent",
                  return_value=_intent_result(Intent.CAUSAL_WHY)),
            patch("app.query.orchestrator.extract_time_range", return_value=None),
            patch("app.query.orchestrator.resolve_entity_filter", return_value=None),
            patch("app.query.orchestrator.causal_planner.run", causal_run),
            patch("app.query.orchestrator.synthesizer.synthesize",
                  return_value=_query_response(Intent.CAUSAL_WHY)),
        ):
            await handle_query(request, pg, neo4j)

        assert causal_run.call_args.kwargs.get("max_hops") == 5
