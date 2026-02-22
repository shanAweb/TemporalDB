"""
app/api/routes/query.py

Natural language query endpoint.

POST /query
    Accept a QueryRequest, run the full Phase 6 pipeline via the
    orchestrator, and return a QueryResponse with a synthesised answer,
    causal chain, supporting events, and source citations.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, status
from neo4j import AsyncSession as Neo4jSession
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import require_api_key
from app.database.neo4j import get_neo4j
from app.database.postgres import get_db
from app.models.schemas.query import QueryRequest, QueryResponse
from app.query.orchestrator import handle_query

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a natural language query",
)
async def query(
    body: QueryRequest,
    pg_session: AsyncSession = Depends(get_db),
    neo4j_session: Neo4jSession = Depends(get_neo4j),
    _key: str = Depends(require_api_key),
) -> QueryResponse:
    """Execute a natural language query against the temporal-causal knowledge base.

    The pipeline classifies intent, extracts temporal constraints, resolves
    entity mentions, dispatches to the appropriate planner, and synthesises
    a cited answer using a local LLM.

    **Intent types**

    | Intent | Example question |
    |---|---|
    | `CAUSAL_WHY` | "Why did revenue drop in Q3?" |
    | `TEMPORAL_RANGE` | "What happened between July and September?" |
    | `SIMILARITY` | "Find events similar to the supply chain disruption" |
    | `ENTITY_TIMELINE` | "Show me everything about Acme Corp" |

    **Optional filters** (applied in addition to NLP inference)

    - `entity_filter` — restrict results to a named entity.
    - `time_range` — explicit UTC start/end window.
    - `max_causal_hops` — depth limit for causal graph traversal (1–10).
    """
    logger.info("query_received", question=body.question, intent_hint=body.entity_filter)

    response = await handle_query(
        request=body,
        pg_session=pg_session,
        neo4j_session=neo4j_session,
    )

    logger.info(
        "query_responded",
        intent=response.intent,
        confidence=response.confidence,
        events=len(response.events),
        sources=len(response.sources),
    )
    return response
