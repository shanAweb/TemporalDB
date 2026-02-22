"""
app/api/routes/entities.py

Entity browsing endpoints.

GET /entities
    Paginated list of entities with optional name substring search and
    type filter.

GET /entities/{entity_id}
    Single entity by UUID.
"""
from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import require_api_key
from app.database.postgres import get_db
from app.models.schemas.entity import EntityListResponse, EntityOut
from app.storage import entity_store

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get(
    "",
    response_model=EntityListResponse,
    summary="List entities",
)
async def list_entities(
    name: str | None = Query(default=None, description="Case-insensitive substring search on name and canonical_name"),
    entity_type: str | None = Query(default=None, description="Filter by NER type (ORG, PERSON, GPE, …)"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results to return"),
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> EntityListResponse:
    """Return a paginated list of entities.

    The optional ``name`` filter matches case-insensitively against both
    the ``name`` and ``canonical_name`` columns using SQL LIKE.  Results
    are ordered by ``canonical_name`` ascending.
    """
    entities, total = await entity_store.list_entities(
        pg_session,
        name_query=name,
        entity_type=entity_type,
        offset=offset,
        limit=limit,
    )

    logger.info(
        "entities_listed",
        total=total,
        returned=len(entities),
        offset=offset,
        limit=limit,
    )
    return EntityListResponse(
        entities=[EntityOut.model_validate(en) for en in entities],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/{entity_id}",
    response_model=EntityOut,
    summary="Get entity by ID",
)
async def get_entity(
    entity_id: uuid.UUID,
    pg_session: AsyncSession = Depends(get_db),
    _key: str = Depends(require_api_key),
) -> EntityOut:
    """Fetch a single entity by its UUID.

    Returns HTTP 404 when no entity with the given ID exists.
    """
    entity = await entity_store.get_entity_by_id(pg_session, entity_id)
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found.",
        )
    logger.info("entity_fetched", entity_id=str(entity_id))
    return EntityOut.model_validate(entity)
