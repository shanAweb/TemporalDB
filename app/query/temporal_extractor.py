"""
app/query/temporal_extractor.py

Extract a UTC time-range constraint from a natural-language query string.

Returns a TimeRange (start, end datetime pair) or None when no temporal
expression can be resolved.  The result is consumed downstream by the
temporal and causal query planners to filter events.

Strategy
--------
1. Pass-through  — if the caller already supplied an explicit TimeRange
   (from QueryRequest.time_range), it is returned immediately without any
   further parsing.

2. NER + temporal parse  — run spaCy NER on the short query string, collect
   DATE / TIME entities, then reuse parse_temporal_entities_sync (same
   logic used by the NLP pipeline) to convert raw text into TemporalSpans.

3. Collapse spans  — fold one or more TemporalSpans into a single TimeRange:
     • 1 span  → use its (ts_start, ts_end) directly.
     • 2 spans → treat as an explicit range: min(ts_start) … max(ts_end).
     • 3+      → take the overall bounding envelope.

4. Relative-expression guard  — dateparser interprets "last month", "Q3",
   etc. relative to *now* at parse time, which is correct for query-time
   use (unlike document ingestion where "now" would be the doc's date).

All CPU-bound work is offloaded to a thread pool so the async event loop
is never blocked.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog

from app.models.schemas.query import TimeRange
from app.nlp.ner import NEREntity, _get_nlp
from app.nlp.temporal_parser import parse_temporal_entities_sync, TemporalSpan

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers (sync, run in thread pool)
# ---------------------------------------------------------------------------

def _extract_ner_entities(question: str) -> list[NEREntity]:
    """Run spaCy NER on *question* and return DATE/TIME entities.

    Reuses the cached spaCy model loaded by the NLP module so there is no
    extra model load cost.

    Args:
        question: Raw query string from the user.

    Returns:
        List of NEREntity objects with label DATE or TIME.
    """
    nlp = _get_nlp()
    doc = nlp(question)
    entities: list[NEREntity] = []
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            entities.append(
                NEREntity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    sentence=question,
                )
            )
    logger.debug(
        "temporal_extractor_ner",
        question=question,
        entity_count=len(entities),
    )
    return entities


def _spans_to_time_range(spans: list[TemporalSpan]) -> TimeRange | None:
    """Collapse one or more TemporalSpans into a single TimeRange.

    Rules:
    - No spans         → None
    - Single point-in-time span → (ts_start, ts_start) to match at least that moment
    - Single range span → (ts_start, ts_end)
    - Multiple spans   → bounding envelope: min(ts_start) … max(ts_end)

    Args:
        spans: List of resolved TemporalSpan objects.

    Returns:
        TimeRange or None.
    """
    if not spans:
        return None

    starts = [s.ts_start for s in spans]
    ends   = [s.ts_end   for s in spans]

    start_dt = min(starts)
    end_dt   = max(ends)

    # Ensure end is not before start (can happen with ambiguous expressions)
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    return TimeRange(start=start_dt, end=end_dt)


def _extract_sync(question: str) -> TimeRange | None:
    """Synchronous end-to-end extraction: NER → parse → collapse."""
    entities = _extract_ner_entities(question)
    if not entities:
        return None

    spans: list[TemporalSpan] = parse_temporal_entities_sync(entities)
    if not spans:
        return None

    result = _spans_to_time_range(spans)
    if result:
        logger.debug(
            "temporal_extractor_resolved",
            start=result.start.isoformat(),
            end=result.end.isoformat(),
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_time_range(
    question: str,
    *,
    explicit: TimeRange | None = None,
) -> TimeRange | None:
    """Extract a UTC time-range constraint from *question*.

    If *explicit* is provided (i.e. the user passed a ``time_range`` in the
    QueryRequest body), it is returned immediately — the caller's explicit
    filter always takes precedence over NLP inference.

    Otherwise the function runs spaCy NER + temporal parsing in a thread
    pool and collapses the results to a single TimeRange.

    Args:
        question: Raw natural-language question from the user.
        explicit: Optional user-supplied TimeRange (pass-through shortcut).

    Returns:
        TimeRange with UTC-aware start and end datetimes, or None when no
        temporal expression could be resolved from the question.
    """
    if explicit is not None:
        logger.debug("temporal_extractor_passthrough")
        return explicit

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _extract_sync, question)
