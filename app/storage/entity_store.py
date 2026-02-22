"""
app/storage/entity_store.py

Entity CRUD across both storage backends (PostgreSQL and Neo4j).

Design
------
The NLP pipeline may encounter the same real-world entity under different
surface forms ("Apple Inc.", "Apple", "AAPL").  This module is responsible
for deduplicating entities by (canonical_name, type) and maintaining an
aliases list of all observed surface forms.

PostgreSQL side
    - canonical_name + type is the logical unique key.
    - upsert_entity() merges surface names into the aliases JSON array.

Neo4j side
    - When a Neo4j session is supplied the corresponding :Entity node is
      kept in sync via graph_store.upsert_entity_node().
    - Passing neo4j_session=None skips the graph write (useful in tests or
      when the graph store is unavailable).

Transaction ownership
    The AsyncSession (Postgres) is flush-only — the caller owns the commit.
    Neo4j writes are auto-committed by the driver within its own session.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import structlog
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from neo4j import AsyncSession as Neo4jSession

from app.models.sql.entity import Entity
from app.storage import graph_store

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_aliases(existing_json: str | None, new_name: str) -> str:
    """Return a JSON array string that includes *new_name*.

    If *new_name* is already present the list is returned unchanged.

    Args:
        existing_json: Current value of Entity.aliases (JSON array or None).
        new_name:      Surface form to add.

    Returns:
        JSON-encoded list of alias strings.
    """
    aliases: list[str] = json.loads(existing_json) if existing_json else []
    if new_name not in aliases:
        aliases.append(new_name)
    return json.dumps(aliases)


# ---------------------------------------------------------------------------
# Core upsert — the primary write path for the NLP pipeline
# ---------------------------------------------------------------------------

async def upsert_entity(
    pg_session: AsyncSession,
    *,
    name: str,
    canonical_name: str,
    entity_type: str,
    description: str | None = None,
    neo4j_session: Neo4jSession | None = None,
) -> Entity:
    """Return the canonical Entity for (canonical_name, entity_type), creating
    it if it does not yet exist, and recording *name* as an alias.

    This is the single entry point used by the NLP pipeline when persisting
    newly linked entities.  It guarantees exactly one Entity row per
    (canonical_name, entity_type) pair across the entire database.

    Steps
    -----
    1. Look up an existing row by ``LOWER(canonical_name)`` and ``type``.
    2. If found: merge *name* into its aliases list and flush.
    3. If not found: insert a fresh row with *name* as the first alias.
    4. If *neo4j_session* is provided: mirror the upsert to the graph store.

    Args:
        pg_session:      SQLAlchemy async session (flush-only).
        name:            Raw surface form observed in the document.
        canonical_name:  Resolved canonical name from the entity linker.
        entity_type:     spaCy NER label (PERSON, ORG, GPE, …).
        description:     Optional free-text description.
        neo4j_session:   Neo4j async session for graph mirror (optional).

    Returns:
        The persisted Entity ORM instance (existing or newly created).
    """
    stmt = select(Entity).where(
        func.lower(Entity.canonical_name) == canonical_name.lower(),
        Entity.type == entity_type,
    )
    result = await pg_session.execute(stmt)
    entity: Entity | None = result.scalar_one_or_none()

    if entity is not None:
        # Merge the new surface form into the existing aliases list.
        entity.aliases = _merge_aliases(entity.aliases, name)
        await pg_session.flush()
        logger.debug(
            "entity_merged",
            entity_id=str(entity.id),
            canonical_name=canonical_name,
        )
    else:
        entity = Entity(
            canonical_name=canonical_name,
            name=name,
            type=entity_type,
            description=description,
            aliases=json.dumps([name]),
        )
        pg_session.add(entity)
        await pg_session.flush()
        await pg_session.refresh(entity)
        logger.debug(
            "entity_created",
            entity_id=str(entity.id),
            canonical_name=canonical_name,
            type=entity_type,
        )

    if neo4j_session is not None:
        await graph_store.upsert_entity_node(
            neo4j_session,
            entity_id=entity.id,
            name=entity.name,
            canonical_name=entity.canonical_name,
            entity_type=entity.type,
        )

    return entity


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

async def get_entity_by_id(
    pg_session: AsyncSession,
    entity_id: uuid.UUID,
    *,
    load_events: bool = False,
) -> Entity | None:
    """Fetch a single entity by primary key.

    Args:
        pg_session:  SQLAlchemy async session.
        entity_id:   Entity UUID.
        load_events: If True, eagerly load the related Event objects via the
                     event_entities join table.

    Returns:
        The Entity instance, or None if not found.
    """
    from sqlalchemy.orm import selectinload

    stmt = select(Entity).where(Entity.id == entity_id)
    if load_events:
        stmt = stmt.options(selectinload(Entity.events))
    result = await pg_session.execute(stmt)
    return result.scalar_one_or_none()


async def get_entity_by_canonical_name(
    pg_session: AsyncSession,
    canonical_name: str,
    entity_type: str,
) -> Entity | None:
    """Look up an entity by its canonical name and type (case-insensitive).

    Args:
        pg_session:     SQLAlchemy async session.
        canonical_name: Canonical entity name to search for.
        entity_type:    spaCy NER label to narrow the lookup.

    Returns:
        The Entity instance, or None if not found.
    """
    stmt = select(Entity).where(
        func.lower(Entity.canonical_name) == canonical_name.lower(),
        Entity.type == entity_type,
    )
    result = await pg_session.execute(stmt)
    return result.scalar_one_or_none()


async def list_entities(
    pg_session: AsyncSession,
    *,
    name_query: str | None = None,
    entity_type: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[Entity], int]:
    """Return a paginated (results, total_count) tuple matching the given filters.

    Filters are optional and combined with AND logic.  The *name_query* is
    matched case-insensitively against both the ``name`` and
    ``canonical_name`` columns using LIKE.

    Results are ordered by canonical_name ascending.

    Args:
        pg_session:  SQLAlchemy async session.
        name_query:  Optional substring to search in name / canonical_name.
        entity_type: Optional NER label to filter by.
        offset:      Number of rows to skip (for pagination).
        limit:       Maximum number of rows to return.

    Returns:
        ``(entities, total_count)`` where total_count reflects filters
        before pagination.
    """
    stmt = select(Entity)

    if name_query:
        pattern = f"%{name_query.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(Entity.name).like(pattern),
                func.lower(Entity.canonical_name).like(pattern),
            )
        )
    if entity_type:
        stmt = stmt.where(Entity.type == entity_type)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await pg_session.execute(count_stmt)).scalar_one()

    data_stmt = (
        stmt
        .order_by(Entity.canonical_name.asc())
        .offset(offset)
        .limit(limit)
    )
    rows = (await pg_session.execute(data_stmt)).scalars().all()
    return list(rows), total


async def get_entities_for_event(
    pg_session: AsyncSession,
    event_id: uuid.UUID,
) -> list[Entity]:
    """Return all entities linked to a given event via event_entities.

    Args:
        pg_session: SQLAlchemy async session.
        event_id:   UUID of the event.

    Returns:
        List of Entity instances (may be empty).
    """
    from app.models.sql.event_entity import EventEntity

    stmt = (
        select(Entity)
        .join(EventEntity, EventEntity.entity_id == Entity.id)
        .where(EventEntity.event_id == event_id)
        .order_by(Entity.canonical_name.asc())
    )
    result = await pg_session.execute(stmt)
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

async def delete_entity(
    pg_session: AsyncSession,
    entity_id: uuid.UUID,
    *,
    neo4j_session: Neo4jSession | None = None,
) -> bool:
    """Delete an entity from PostgreSQL and optionally from Neo4j.

    The ON DELETE CASCADE constraint on event_entities handles join-table
    cleanup automatically.

    Args:
        pg_session:    SQLAlchemy async session (flush-only).
        entity_id:     UUID of the entity to delete.
        neo4j_session: Neo4j session for graph mirror (optional).

    Returns:
        True if the entity existed and was deleted, False otherwise.
    """
    stmt = (
        delete(Entity)
        .where(Entity.id == entity_id)
        .returning(Entity.id)
    )
    result = await pg_session.execute(stmt)
    await pg_session.flush()
    deleted = result.scalar_one_or_none() is not None

    if deleted and neo4j_session is not None:
        # Entity nodes in Neo4j have no dedicated delete function in
        # graph_store, so we issue a targeted DETACH DELETE directly.
        cypher = "MATCH (en:Entity {id: $id}) DETACH DELETE en"
        await neo4j_session.run(cypher, {"id": str(entity_id)})

    logger.debug("entity_deleted", entity_id=str(entity_id), found=deleted)
    return deleted


# ---------------------------------------------------------------------------
# Bulk helpers (used by the NLP pipeline and sync.py)
# ---------------------------------------------------------------------------

async def bulk_upsert_entities(
    pg_session: AsyncSession,
    entities: list[dict[str, Any]],
    *,
    neo4j_session: Neo4jSession | None = None,
) -> list[Entity]:
    """Upsert a batch of entities in a single pass, returning persisted objects.

    Each dict in *entities* must contain the keys accepted by upsert_entity():
    ``name``, ``canonical_name``, ``entity_type``.  ``description`` is optional.

    Entities sharing the same (canonical_name, entity_type) within the batch
    are collapsed to a single row — only the first occurrence's description is
    used; subsequent names are merged into aliases.

    Args:
        pg_session:   SQLAlchemy async session (flush-only).
        entities:     List of entity dicts (see upsert_entity for keys).
        neo4j_session: Neo4j session for graph mirror (optional).

    Returns:
        List of Entity instances in the same order as the input.
    """
    results: list[Entity] = []
    # Cache within this batch to avoid redundant DB round-trips for the
    # same canonical entity appearing multiple times in one document.
    cache: dict[tuple[str, str], Entity] = {}

    for spec in entities:
        key = (spec["canonical_name"].lower(), spec["entity_type"])
        if key in cache:
            # Already upserted in this batch — just merge the alias.
            cached = cache[key]
            cached.aliases = _merge_aliases(cached.aliases, spec["name"])
            results.append(cached)
            continue

        entity = await upsert_entity(
            pg_session,
            name=spec["name"],
            canonical_name=spec["canonical_name"],
            entity_type=spec["entity_type"],
            description=spec.get("description"),
            neo4j_session=neo4j_session,
        )
        cache[key] = entity
        results.append(entity)

    logger.debug("bulk_entity_upsert_done", count=len(results))
    return results
