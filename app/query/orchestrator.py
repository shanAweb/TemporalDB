"""
app/query/orchestrator.py

Main query handler — wires every Phase 6 component into a single pipeline.

Pipeline
--------
  QueryRequest
      │
      ├─► classify_intent()          → Intent
      ├─► extract_time_range()       → TimeRange | None
      ├─► resolve_entity_filter()    → UUID | None
      │
      ├─► dispatch to planner ──────────────────────────────────────┐
      │     CAUSAL_WHY      → causal_planner.run()                  │
      │     TEMPORAL_RANGE  → temporal_planner.run()                │
      │     SIMILARITY      → similarity_planner.run()              │
      │     ENTITY_TIMELINE → entity_planner.run()                  │
      │                                                             │
      └─► synthesize(plan)  ◄───────────────────────────────────────┘
              │
              └─► QueryResponse

Sessions
--------
Both the PostgreSQL AsyncSession and the Neo4j AsyncSession are injected
by the caller (FastAPI route via Depends).  The orchestrator is stateless
and does not own transaction boundaries.
"""
from __future__ import annotations

import structlog
from neo4j import AsyncSession as Neo4jSession
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas.query import QueryRequest, QueryResponse
from app.query import synthesizer
from app.query.entity_resolver import resolve_entity_filter
from app.query.intent import Intent, classify_intent
from app.query.planners import PlanResult
from app.query.planners import (
    causal_planner,
    entity_planner,
    similarity_planner,
    temporal_planner,
)
from app.query.temporal_extractor import extract_time_range

logger = structlog.get_logger(__name__)


async def handle_query(
    request: QueryRequest,
    pg_session: AsyncSession,
    neo4j_session: Neo4jSession,
) -> QueryResponse:
    """Execute the full query pipeline for *request*.

    Args:
        request:       Validated QueryRequest from the API route.
        pg_session:    SQLAlchemy async session (read-only within this call).
        neo4j_session: Neo4j async session.

    Returns:
        QueryResponse with answer, citations, events, and causal chain.
    """
    question = request.question

    # ── Stage 1: classify intent ───────────────────────────────────────────
    intent_result = await classify_intent(question)
    intent = intent_result.intent
    logger.info(
        "query_intent_classified",
        intent=intent.value,
        confidence=intent_result.confidence,
        method=intent_result.method,
    )

    # ── Stage 2: extract time range ────────────────────────────────────────
    time_range = await extract_time_range(
        question,
        explicit=request.time_range,
    )
    logger.info(
        "query_time_range",
        resolved=time_range is not None,
        start=time_range.start.isoformat() if time_range else None,
        end=time_range.end.isoformat()     if time_range else None,
    )

    # ── Stage 3: resolve entity filter ────────────────────────────────────
    entity_id = await resolve_entity_filter(pg_session, request.entity_filter)
    logger.info(
        "query_entity_resolved",
        filter=request.entity_filter,
        entity_id=str(entity_id) if entity_id else None,
    )

    # ── Stage 4: dispatch to the appropriate planner ──────────────────────
    plan: PlanResult

    if intent == Intent.CAUSAL_WHY:
        plan = await causal_planner.run(
            pg_session,
            neo4j_session,
            question,
            entity_id=entity_id,
            max_hops=request.max_causal_hops,
        )

    elif intent == Intent.TEMPORAL_RANGE:
        plan = await temporal_planner.run(
            pg_session,
            time_range,
            entity_id=entity_id,
        )

    elif intent == Intent.ENTITY_TIMELINE:
        plan = await entity_planner.run(
            pg_session,
            neo4j_session,
            entity_id,
            time_range=time_range,
        )

    else:  # SIMILARITY (default)
        plan = await similarity_planner.run(
            pg_session,
            question,
            entity_id=entity_id,
            time_range=time_range,
        )

    logger.info(
        "query_plan_complete",
        intent=intent.value,
        events=len(plan.events),
        chain_nodes=len(plan.causal_chain),
        planner_confidence=plan.confidence,
    )

    # ── Stage 5: synthesize answer ─────────────────────────────────────────
    response = await synthesizer.synthesize(
        pg_session,
        plan,
        question,
        intent,
    )

    logger.info(
        "query_complete",
        intent=intent.value,
        answer_length=len(response.answer),
        confidence=response.confidence,
    )
    return response
