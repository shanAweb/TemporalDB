"""Temporal expression parsing and UTC normalization.

Extracts DATE and TIME entities from text using spaCy NER, then parses
each expression into UTC-normalized datetime objects using dateparser.
Range expressions (quarters, months, years) are expanded into
(start, end) pairs.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import dateparser
import structlog

from app.nlp.ner import NEREntity

logger = structlog.get_logger(__name__)

# dateparser settings applied to every parse call
_DATEPARSER_SETTINGS: dict[str, Any] = {
    "RETURN_AS_TIMEZONE_AWARE": True,
    "PREFER_DAY_OF_MONTH": "first",
    "PREFER_DATES_FROM": "past",
    "TO_TIMEZONE": "UTC",
}

# Quarter → (month_start, month_end) mapping
_QUARTER_MAP = {
    "q1": (1, 3),
    "q2": (4, 6),
    "q3": (7, 9),
    "q4": (10, 12),
}


@dataclass
class TemporalSpan:
    """A resolved temporal expression from the document."""

    text: str               # Original text (e.g. "Q3 2024", "last month")
    ts_start: datetime      # UTC start of the period
    ts_end: datetime        # UTC end of the period (same as ts_start for point-in-time)
    is_range: bool = False  # True when the expression covers a period


def _to_utc(dt: datetime) -> datetime:
    """Ensure a datetime is UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_quarter(text: str, reference_year: int | None = None) -> TemporalSpan | None:
    """Detect and expand quarter expressions like 'Q3 2024' or 'q1'."""
    lowered = text.lower().replace(" ", "")
    year = reference_year or datetime.now(timezone.utc).year

    for quarter_key, (m_start, m_end) in _QUARTER_MAP.items():
        if quarter_key in lowered:
            # Try to extract a 4-digit year from the original text
            import re  # noqa: PLC0415
            year_match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
            if year_match:
                year = int(year_match.group(1))

            ts_start = datetime(year, m_start, 1, tzinfo=timezone.utc)
            # Last day of end month
            import calendar  # noqa: PLC0415
            last_day = calendar.monthrange(year, m_end)[1]
            ts_end = datetime(year, m_end, last_day, 23, 59, 59, tzinfo=timezone.utc)

            return TemporalSpan(text=text, ts_start=ts_start, ts_end=ts_end, is_range=True)
    return None


def _parse_expression(text: str) -> TemporalSpan | None:
    """Parse a single temporal expression into a TemporalSpan.

    Tries quarter detection first, then falls back to dateparser.

    Args:
        text: Raw temporal expression string.

    Returns:
        TemporalSpan or None if unparseable.
    """
    # 1. Quarter expressions
    quarter = _parse_quarter(text)
    if quarter:
        return quarter

    # 2. dateparser for everything else
    parsed = dateparser.parse(text, settings=_DATEPARSER_SETTINGS)
    if parsed is None:
        logger.debug("temporal_parse_failed", expression=text)
        return None

    ts = _to_utc(parsed)

    # Treat year-only or month-only expressions as ranges
    lowered = text.lower().strip()
    import re  # noqa: PLC0415

    # Year only: e.g. "2024"
    if re.fullmatch(r"(19|20)\d{2}", lowered):
        year = int(lowered)
        ts_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        ts_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return TemporalSpan(text=text, ts_start=ts_start, ts_end=ts_end, is_range=True)

    # Month + year: e.g. "July 2024"
    import calendar  # noqa: PLC0415
    month_year = re.fullmatch(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(19|20)\d{2}",
        lowered,
    )
    if month_year:
        last_day = calendar.monthrange(ts.year, ts.month)[1]
        ts_start = ts.replace(day=1, hour=0, minute=0, second=0)
        ts_end = ts.replace(day=last_day, hour=23, minute=59, second=59)
        return TemporalSpan(text=text, ts_start=ts_start, ts_end=ts_end, is_range=True)

    return TemporalSpan(text=text, ts_start=ts, ts_end=ts, is_range=False)


def parse_temporal_entities_sync(entities: list[NEREntity]) -> list[TemporalSpan]:
    """Convert DATE/TIME NEREntity objects into normalized TemporalSpans.

    Args:
        entities: List of NEREntity objects from the NER module.

    Returns:
        List of TemporalSpan objects for entities with label DATE or TIME.
    """
    spans: list[TemporalSpan] = []

    for ent in entities:
        if ent.label not in ("DATE", "TIME"):
            continue

        span = _parse_expression(ent.text)
        if span:
            spans.append(span)
        else:
            logger.warning("temporal_entity_unparseable", text=ent.text)

    logger.debug("temporal_spans_parsed", count=len(spans))
    return spans


async def parse_temporal_entities(entities: list[NEREntity]) -> list[TemporalSpan]:
    """Async wrapper for parse_temporal_entities_sync.

    Args:
        entities: List of NEREntity objects from the NER module.

    Returns:
        List of resolved TemporalSpan objects.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, parse_temporal_entities_sync, entities)
