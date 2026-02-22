"""
app/query/synthesizer.py

Answer synthesis with structured citations.

Converts a PlanResult (raw ORM objects + Neo4j chain records) into a
fully-formed QueryResponse by:

  1. Fetching source Document rows from PostgreSQL for the document IDs
     referenced by the planner's events.

  2. Converting ORM objects to Pydantic schemas:
       Event        → EventBrief
       chain record → CausalChainLink
       Document     → SourceReference

  3. Formatting events, causal chain, and sources into a prompt and calling
     Ollama (llama3.1:8b) to generate a natural-language answer.

  4. Falling back to a structured template answer when Ollama is unavailable
     so the API always returns a usable response.

The synthesizer is intentionally stateless — it does not write to any store.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.client import ollama_client
from app.llm.prompts import ANSWER_SYNTHESIS
from app.models.schemas.event import EventBrief
from app.models.schemas.query import CausalChainLink, QueryResponse, SourceReference
from app.models.sql.document import Document
from app.query.intent import Intent
from app.query.planners import PlanResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal conversion helpers
# ---------------------------------------------------------------------------

def _events_to_brief(plan: PlanResult) -> list[EventBrief]:
    """Convert PlanResult.events (ORM) to EventBrief Pydantic objects."""
    briefs: list[EventBrief] = []
    for ev in plan.events:
        briefs.append(
            EventBrief(
                id=ev.id,
                description=ev.description,
                ts_start=ev.ts_start,
                confidence=ev.confidence,
            )
        )
    return briefs


def _chain_to_links(plan: PlanResult) -> list[CausalChainLink]:
    """Convert PlanResult.causal_chain (Neo4j dicts) to CausalChainLink objects.

    Each dict is expected to have at minimum:
        event_id    — string UUID
        description — str
        ts_start    — ISO-8601 string or None
        confidence  — float
    """
    links: list[CausalChainLink] = []
    for record in plan.causal_chain:
        try:
            event_id = uuid.UUID(record["event_id"])
        except (KeyError, ValueError):
            continue

        raw_ts = record.get("ts_start")
        ts_start: datetime | None = None
        if raw_ts:
            try:
                ts_start = datetime.fromisoformat(raw_ts)
            except ValueError:
                pass

        links.append(
            CausalChainLink(
                id=event_id,
                description=record.get("description", ""),
                ts_start=ts_start,
                confidence=float(record.get("confidence", 1.0)),
            )
        )
    return links


async def _fetch_sources(
    pg_session: AsyncSession,
    document_ids: set[uuid.UUID],
) -> list[SourceReference]:
    """Fetch Document rows and convert to SourceReference objects.

    Args:
        pg_session:   SQLAlchemy async session.
        document_ids: Set of document UUIDs to fetch.

    Returns:
        List of SourceReference objects, one per resolved document.
    """
    if not document_ids:
        return []

    stmt = select(Document).where(Document.id.in_(document_ids))
    docs = list((await pg_session.execute(stmt)).scalars().all())

    sources: list[SourceReference] = []
    for doc in docs:
        meta: dict | None = None
        if doc.metadata_:
            try:
                meta = json.loads(doc.metadata_)
            except (ValueError, TypeError):
                pass
        if doc.filename:
            meta = (meta or {}) | {"filename": doc.filename}

        sources.append(
            SourceReference(
                id=doc.id,
                source=doc.source,
                metadata=meta,
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_events(briefs: list[EventBrief]) -> str:
    if not briefs:
        return "No events retrieved."
    lines = []
    for ev in briefs:
        ts = ev.ts_start.isoformat() if ev.ts_start else "unknown date"
        lines.append(f"- [{ts}] {ev.description} (confidence: {ev.confidence:.2f})")
    return "\n".join(lines)


def _format_chain(links: list[CausalChainLink]) -> str:
    if not links:
        return "No causal chain available."
    lines = []
    for i, link in enumerate(links, 1):
        ts = link.ts_start.isoformat() if link.ts_start else "unknown date"
        lines.append(
            f"{i}. [{ts}] {link.description} (confidence: {link.confidence:.2f})"
        )
    return "\n".join(lines)


def _format_sources(sources: list[SourceReference]) -> str:
    if not sources:
        return "No source documents."
    lines = [f"- {src.source} (id: {src.id})" for src in sources]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fallback answer (when LLM is unavailable)
# ---------------------------------------------------------------------------

def _fallback_answer(
    question: str,
    briefs: list[EventBrief],
    links: list[CausalChainLink],
) -> str:
    """Build a minimal structured answer without LLM synthesis."""
    parts = [f"Query: {question}\n"]

    if links:
        parts.append("Causal chain identified:")
        for i, link in enumerate(links, 1):
            ts = link.ts_start.isoformat() if link.ts_start else "unknown date"
            parts.append(f"  {i}. {link.description} ({ts})")
    elif briefs:
        parts.append("Relevant events found:")
        for ev in briefs[:5]:
            ts = ev.ts_start.isoformat() if ev.ts_start else "unknown date"
            parts.append(f"  - {ev.description} ({ts})")
    else:
        parts.append("No relevant events found for this query.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def synthesize(
    pg_session: AsyncSession,
    plan: PlanResult,
    question: str,
    intent: Intent,
) -> QueryResponse:
    """Synthesize a cited natural-language answer from a PlanResult.

    Calls Ollama to generate the prose answer, with a structured fallback
    if the LLM is unavailable.

    Args:
        pg_session: SQLAlchemy async session (read-only; used for doc fetch).
        plan:       PlanResult from any of the four query planners.
        question:   Original user question.
        intent:     Classified intent (included verbatim in the response).

    Returns:
        Fully-populated QueryResponse ready to return from the API.
    """
    # Convert raw planner output to Pydantic schemas.
    briefs  = _events_to_brief(plan)
    links   = _chain_to_links(plan)
    sources = await _fetch_sources(pg_session, plan.document_ids)

    # Build the Ollama prompt.
    prompt = ANSWER_SYNTHESIS.format(
        question=question,
        events=_format_events(briefs),
        causal_chain=_format_chain(links),
        sources=_format_sources(sources),
    )

    # Call Ollama; fall back to a template answer on any failure.
    try:
        answer = await ollama_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
        )
        answer = answer.strip()
        logger.info("synthesizer_llm_answer", question=question, length=len(answer))
    except Exception as exc:  # noqa: BLE001
        logger.warning("synthesizer_llm_failed", error=str(exc))
        answer = _fallback_answer(question, briefs, links)

    return QueryResponse(
        answer=answer,
        confidence=round(plan.confidence, 4),
        intent=intent.value,
        causal_chain=links,
        events=briefs,
        sources=sources,
    )
