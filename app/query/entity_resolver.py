"""
app/query/entity_resolver.py

Resolve a free-text entity mention (from a user query) to a canonical
Entity UUID stored in PostgreSQL.

Used by every query planner that needs to filter or anchor results on a
specific entity (e.g. "Acme Corp", "John Smith", "Q3 revenue drop").

Resolution strategy  (three tiers, short-circuit on first match)
-----------------------------------------------------------------
Tier 1 — Exact match
    Case-insensitive equality on ``canonical_name`` or ``name``.
    Confidence: 1.0.

Tier 2 — Alias match
    Load the top ILIKE candidates (name / canonical_name LIKE %mention%),
    then check each entity's JSON aliases array for the mention.  This
    catches surface forms observed during ingestion that differ from the
    canonical name (e.g. "Apple" matching a canonical entity "Apple Inc.").
    Confidence: 0.95.

Tier 3 — Fuzzy string match
    Apply SequenceMatcher to the ILIKE candidate set.  Returns the best
    match above ``_FUZZY_THRESHOLD`` (default 0.75).
    Confidence: SequenceMatcher ratio.

Returns None if no candidate clears the minimum confidence threshold.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher

import structlog
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sql.entity import Entity

logger = structlog.get_logger(__name__)

# Minimum SequenceMatcher ratio to accept a fuzzy match.
_FUZZY_THRESHOLD: float = 0.75

# Maximum ILIKE candidates to load for alias/fuzzy checking.
_CANDIDATE_LIMIT: int = 20


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ResolveResult:
    """A resolved entity mention."""

    entity_id: uuid.UUID
    canonical_name: str
    entity_type: str
    confidence: float       # 0.0 – 1.0
    method: str             # "exact" | "alias" | "fuzzy"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return text.lower().strip()


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _check_aliases(entity: Entity, mention: str) -> bool:
    """Return True if *mention* appears (case-insensitively) in entity.aliases."""
    if not entity.aliases:
        return False
    try:
        aliases: list[str] = json.loads(entity.aliases)
    except (ValueError, TypeError):
        return False
    norm_mention = _norm(mention)
    return any(_norm(alias) == norm_mention for alias in aliases)


async def _load_candidates(
    pg_session: AsyncSession,
    mention: str,
    entity_type: str | None,
) -> list[Entity]:
    """Fetch entities whose name or canonical_name contains *mention* (ILIKE).

    Also includes an exact-match clause so Tier 1 hits are always in the
    candidate set even when the ILIKE clause would miss them.

    Args:
        pg_session:  SQLAlchemy async session.
        mention:     Raw mention text from the user query.
        entity_type: Optional NER label to narrow the search.

    Returns:
        Up to _CANDIDATE_LIMIT Entity rows, ordered by canonical_name.
    """
    pattern = f"%{mention}%"
    norm_mention = mention.lower()

    stmt = select(Entity).where(
        or_(
            func.lower(Entity.canonical_name).like(f"%{norm_mention}%"),
            func.lower(Entity.name).like(f"%{norm_mention}%"),
        )
    )
    if entity_type:
        stmt = stmt.where(Entity.type == entity_type)

    stmt = stmt.order_by(Entity.canonical_name.asc()).limit(_CANDIDATE_LIMIT)
    result = await pg_session.execute(stmt)
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def resolve_entity(
    pg_session: AsyncSession,
    mention: str,
    *,
    entity_type: str | None = None,
) -> ResolveResult | None:
    """Resolve *mention* to a canonical Entity UUID using a three-tier strategy.

    Args:
        pg_session:  SQLAlchemy async session.
        mention:     Free-text entity name from the user's query
                     (e.g. ``"Acme Corp"``, ``"John Smith"``).
        entity_type: Optional spaCy NER label (``"ORG"``, ``"PERSON"``, …)
                     to restrict matching to a single entity type.

    Returns:
        ResolveResult with the best matching entity, or None if no match
        clears the minimum confidence threshold.
    """
    if not mention or not mention.strip():
        return None

    mention = mention.strip()
    candidates = await _load_candidates(pg_session, mention, entity_type)

    if not candidates:
        logger.debug("entity_resolver_no_candidates", mention=mention)
        return None

    norm_mention = _norm(mention)

    # ── Tier 1: Exact match ────────────────────────────────────────────────
    for entity in candidates:
        if (
            _norm(entity.canonical_name) == norm_mention
            or _norm(entity.name) == norm_mention
        ):
            logger.debug(
                "entity_resolver_exact",
                mention=mention,
                entity_id=str(entity.id),
            )
            return ResolveResult(
                entity_id=entity.id,
                canonical_name=entity.canonical_name,
                entity_type=entity.type,
                confidence=1.0,
                method="exact",
            )

    # ── Tier 2: Alias match ────────────────────────────────────────────────
    for entity in candidates:
        if _check_aliases(entity, mention):
            logger.debug(
                "entity_resolver_alias",
                mention=mention,
                entity_id=str(entity.id),
            )
            return ResolveResult(
                entity_id=entity.id,
                canonical_name=entity.canonical_name,
                entity_type=entity.type,
                confidence=0.95,
                method="alias",
            )

    # ── Tier 3: Fuzzy string match ─────────────────────────────────────────
    best_entity: Entity | None = None
    best_ratio: float = 0.0

    for entity in candidates:
        # Score against both canonical_name and name; take the higher.
        ratio = max(
            _fuzzy_ratio(mention, entity.canonical_name),
            _fuzzy_ratio(mention, entity.name),
        )
        if ratio > best_ratio:
            best_ratio = ratio
            best_entity = entity

    if best_entity is not None and best_ratio >= _FUZZY_THRESHOLD:
        logger.debug(
            "entity_resolver_fuzzy",
            mention=mention,
            entity_id=str(best_entity.id),
            ratio=round(best_ratio, 3),
        )
        return ResolveResult(
            entity_id=best_entity.id,
            canonical_name=best_entity.canonical_name,
            entity_type=best_entity.type,
            confidence=round(best_ratio, 4),
            method="fuzzy",
        )

    logger.debug(
        "entity_resolver_no_match",
        mention=mention,
        best_ratio=round(best_ratio, 3),
    )
    return None


async def resolve_entity_filter(
    pg_session: AsyncSession,
    entity_filter: str | None,
) -> uuid.UUID | None:
    """Convenience wrapper used by planners to resolve QueryRequest.entity_filter.

    Returns the entity UUID if the mention resolves successfully, or None
    when no filter is set or resolution fails.

    Args:
        pg_session:    SQLAlchemy async session.
        entity_filter: Optional entity mention string from QueryRequest.

    Returns:
        Entity UUID or None.
    """
    if not entity_filter:
        return None

    result = await resolve_entity(pg_session, entity_filter)
    if result is None:
        logger.warning("entity_filter_unresolved", mention=entity_filter)
        return None

    return result.entity_id
