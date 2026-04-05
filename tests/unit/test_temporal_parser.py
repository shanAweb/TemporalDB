"""
tests/unit/test_temporal_parser.py

Unit tests for app.nlp.temporal_parser.

Tests the pure helper functions (_parse_quarter, _parse_expression) and
the public parse_temporal_entities_sync() function.  No ML models are
loaded — dateparser is used but is a pure-Python library.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.nlp.ner import NEREntity
from app.nlp.temporal_parser import (
    TemporalSpan,
    _parse_expression,
    _parse_quarter,
    parse_temporal_entities_sync,
)


# ---------------------------------------------------------------------------
# _parse_quarter
# ---------------------------------------------------------------------------

class TestParseQuarter:
    def test_q3_with_year(self) -> None:
        span = _parse_quarter("Q3 2024")
        assert span is not None
        assert span.ts_start == datetime(2024, 7, 1, tzinfo=timezone.utc)
        assert span.ts_end == datetime(2024, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
        assert span.is_range is True

    def test_q1_with_year(self) -> None:
        span = _parse_quarter("Q1 2023")
        assert span is not None
        assert span.ts_start == datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert span.ts_end == datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_q4_with_year(self) -> None:
        span = _parse_quarter("Q4 2022")
        assert span is not None
        assert span.ts_start.month == 10
        assert span.ts_end.month == 12

    def test_lowercase_q(self) -> None:
        span = _parse_quarter("q2 2024")
        assert span is not None
        assert span.ts_start.month == 4

    def test_no_year_uses_current(self) -> None:
        span = _parse_quarter("Q2")
        assert span is not None
        current_year = datetime.now(timezone.utc).year
        assert span.ts_start.year == current_year

    def test_non_quarter_returns_none(self) -> None:
        assert _parse_quarter("July 2024") is None
        assert _parse_quarter("2024") is None
        assert _parse_quarter("last month") is None


# ---------------------------------------------------------------------------
# _parse_expression
# ---------------------------------------------------------------------------

class TestParseExpression:
    def test_year_only(self) -> None:
        span = _parse_expression("2024")
        assert span is not None
        assert span.ts_start == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert span.ts_end == datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        assert span.is_range is True

    def test_month_year(self) -> None:
        span = _parse_expression("July 2024")
        assert span is not None
        assert span.ts_start.month == 7
        assert span.ts_start.day == 1
        assert span.ts_end.month == 7
        assert span.ts_end.day == 31
        assert span.is_range is True

    def test_quarter_via_expression(self) -> None:
        span = _parse_expression("Q3 2024")
        assert span is not None
        assert span.ts_start.month == 7
        assert span.is_range is True

    def test_garbage_returns_none(self) -> None:
        # dateparser may still parse some vague strings, but pure garbage should fail
        span = _parse_expression("xyzzy foobar 99999")
        assert span is None

    def test_result_is_utc_aware(self) -> None:
        span = _parse_expression("2023")
        assert span is not None
        assert span.ts_start.tzinfo is not None
        assert span.ts_end.tzinfo is not None


# ---------------------------------------------------------------------------
# parse_temporal_entities_sync
# ---------------------------------------------------------------------------

def _make_entity(text: str, label: str = "DATE") -> NEREntity:
    return NEREntity(
        text=text,
        label=label,
        start_char=0,
        end_char=len(text),
        sentence=text,
    )


class TestParseTemporalEntitiesSync:
    def test_filters_non_date_entities(self) -> None:
        entities = [
            _make_entity("Apple Inc.", label="ORG"),
            _make_entity("John Smith", label="PERSON"),
        ]
        spans = parse_temporal_entities_sync(entities)
        assert spans == []

    def test_parses_date_entities(self) -> None:
        entities = [_make_entity("Q3 2024", label="DATE")]
        spans = parse_temporal_entities_sync(entities)
        assert len(spans) == 1
        assert spans[0].ts_start.month == 7

    def test_multiple_dates(self) -> None:
        entities = [
            _make_entity("2023", label="DATE"),
            _make_entity("2024", label="DATE"),
        ]
        spans = parse_temporal_entities_sync(entities)
        assert len(spans) == 2
        years = {s.ts_start.year for s in spans}
        assert years == {2023, 2024}

    def test_mixed_labels(self) -> None:
        entities = [
            _make_entity("Acme Corp", label="ORG"),
            _make_entity("Q1 2023", label="DATE"),
            _make_entity("10:00 AM", label="TIME"),
        ]
        spans = parse_temporal_entities_sync(entities)
        # DATE and TIME entities are both processed
        assert len(spans) >= 1

    def test_empty_input(self) -> None:
        assert parse_temporal_entities_sync([]) == []
