"""
tests/unit/test_temporal_extractor.py

Unit tests for app.query.temporal_extractor.

Tests the pure _spans_to_time_range() helper synchronously, and tests
extract_time_range() by patching _extract_sync so no spaCy model is
loaded.

Coverage
--------
  - _spans_to_time_range: empty list → None
  - _spans_to_time_range: single span → (ts_start, ts_end)
  - _spans_to_time_range: multiple spans → bounding envelope
  - _spans_to_time_range: end < start → swapped
  - extract_time_range: explicit TimeRange returned without NLP
  - extract_time_range: NLP path returns result from _extract_sync
  - extract_time_range: NLP path returns None when no temporal expression
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.models.schemas.query import TimeRange
from app.nlp.temporal_parser import TemporalSpan
from app.query.temporal_extractor import _spans_to_time_range, extract_time_range


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span(
    year_start: int,
    year_end: int | None = None,
    month_start: int = 1,
    month_end: int = 12,
) -> TemporalSpan:
    ts_start = datetime(year_start, month_start, 1, tzinfo=timezone.utc)
    ts_end   = datetime(year_end or year_start, month_end, 28, tzinfo=timezone.utc)
    return TemporalSpan(
        text=f"{year_start}",
        ts_start=ts_start,
        ts_end=ts_end,
        is_range=(ts_start != ts_end),
    )


def _tr(start_year: int, end_year: int) -> TimeRange:
    return TimeRange(
        start=datetime(start_year, 1, 1, tzinfo=timezone.utc),
        end=datetime(end_year, 12, 31, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# _spans_to_time_range
# ---------------------------------------------------------------------------

class TestSpansToTimeRange:
    def test_empty_list_returns_none(self) -> None:
        assert _spans_to_time_range([]) is None

    def test_single_span_returns_its_bounds(self) -> None:
        span = _span(2024, month_start=7, month_end=9)
        result = _spans_to_time_range([span])
        assert result is not None
        assert result.start == span.ts_start
        assert result.end   == span.ts_end

    def test_two_spans_bounding_envelope(self) -> None:
        s1 = _span(2023)
        s2 = _span(2024)
        result = _spans_to_time_range([s1, s2])
        assert result is not None
        assert result.start == s1.ts_start   # min(start)
        assert result.end   == s2.ts_end     # max(end)

    def test_three_spans_envelope(self) -> None:
        spans = [_span(2022), _span(2023), _span(2024)]
        result = _spans_to_time_range(spans)
        assert result is not None
        assert result.start.year == 2022
        assert result.end.year   == 2024

    def test_end_before_start_gets_swapped(self) -> None:
        # Construct spans where min(ts_start) > max(ts_end), which can happen
        # with ambiguous date expressions.  The function should swap them.
        early_ts = datetime(2022, 1, 1, tzinfo=timezone.utc)
        late_ts  = datetime(2024, 12, 31, tzinfo=timezone.utc)
        # ts_start intentionally larger than ts_end to trigger the swap.
        span = TemporalSpan(
            text="ambiguous",
            ts_start=late_ts,
            ts_end=early_ts,
            is_range=True,
        )
        result = _spans_to_time_range([span])
        assert result is not None
        assert result.start <= result.end

    def test_returns_time_range_type(self) -> None:
        span = _span(2024)
        result = _spans_to_time_range([span])
        assert isinstance(result, TimeRange)

    def test_result_preserves_timezone(self) -> None:
        span = _span(2024)
        result = _spans_to_time_range([span])
        assert result is not None
        assert result.start.tzinfo is not None
        assert result.end.tzinfo   is not None


# ---------------------------------------------------------------------------
# extract_time_range
# ---------------------------------------------------------------------------

class TestExtractTimeRange:
    @pytest.mark.asyncio
    async def test_explicit_passthrough_without_nlp(self) -> None:
        """Explicit TimeRange must be returned immediately; _extract_sync never called."""
        explicit = _tr(2024, 2024)
        with patch("app.query.temporal_extractor._extract_sync") as mock_extract:
            result = await extract_time_range("any question", explicit=explicit)
        mock_extract.assert_not_called()
        assert result is explicit

    @pytest.mark.asyncio
    async def test_explicit_none_calls_nlp(self) -> None:
        """Without explicit TimeRange, _extract_sync should be invoked."""
        expected = _tr(2023, 2024)
        with patch(
            "app.query.temporal_extractor._extract_sync",
            return_value=expected,
        ):
            result = await extract_time_range("What happened in 2023 and 2024?")
        assert result is expected

    @pytest.mark.asyncio
    async def test_nlp_returns_none_when_no_expression(self) -> None:
        """When _extract_sync finds nothing, None is returned."""
        with patch(
            "app.query.temporal_extractor._extract_sync",
            return_value=None,
        ):
            result = await extract_time_range("Tell me about events.")
        assert result is None

    @pytest.mark.asyncio
    async def test_explicit_takes_precedence_over_nlp(self) -> None:
        """Even when question contains dates, explicit is used if provided."""
        explicit = _tr(2024, 2024)
        nlp_result = _tr(2020, 2020)
        with patch(
            "app.query.temporal_extractor._extract_sync",
            return_value=nlp_result,
        ):
            result = await extract_time_range(
                "What happened in 2020?", explicit=explicit
            )
        # Explicit must win.
        assert result is explicit
        assert result.start.year == 2024
