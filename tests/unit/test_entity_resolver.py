"""
tests/unit/test_entity_resolver.py

Unit tests for app.query.entity_resolver.

Tests the pure helpers (_norm, _fuzzy_ratio, _check_aliases) and the
async resolve_entity() / resolve_entity_filter() functions with a
mocked SQLAlchemy AsyncSession so no database is required.

Coverage
--------
  - _norm: lowercases and strips whitespace
  - _fuzzy_ratio: symmetric, ≈1.0 for identical, <threshold for disjoint
  - _check_aliases: JSON parse, case-insensitive match, missing/invalid
  - resolve_entity: Tier 1 exact match (confidence 1.0)
  - resolve_entity: Tier 2 alias match (confidence 0.95)
  - resolve_entity: Tier 3 fuzzy match above threshold
  - resolve_entity: returns None when no candidates
  - resolve_entity: returns None when fuzzy ratio below threshold
  - resolve_entity: empty/blank mention returns None immediately
  - resolve_entity_filter: None filter returns None, resolved returns UUID
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.query.entity_resolver import (
    _FUZZY_THRESHOLD,
    _check_aliases,
    _fuzzy_ratio,
    _norm,
    resolve_entity,
    resolve_entity_filter,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestNorm:
    def test_lowercases(self) -> None:
        assert _norm("Apple Inc.") == "apple inc."

    def test_strips_whitespace(self) -> None:
        assert _norm("  Acme Corp  ") == "acme corp"

    def test_empty_string(self) -> None:
        assert _norm("") == ""


class TestFuzzyRatio:
    def test_identical_strings(self) -> None:
        assert _fuzzy_ratio("Apple Inc.", "Apple Inc.") == pytest.approx(1.0)

    def test_symmetric(self) -> None:
        assert _fuzzy_ratio("Apple", "Banana") == pytest.approx(
            _fuzzy_ratio("Banana", "Apple")
        )

    def test_completely_different(self) -> None:
        assert _fuzzy_ratio("Apple Inc.", "Zeta Corp") < _FUZZY_THRESHOLD

    def test_case_insensitive(self) -> None:
        assert _fuzzy_ratio("APPLE INC.", "apple inc.") == pytest.approx(1.0)


class TestCheckAliases:
    def _entity(self, aliases: list[str] | None) -> SimpleNamespace:
        return SimpleNamespace(
            aliases=json.dumps(aliases) if aliases is not None else None
        )

    def test_match_found(self) -> None:
        ent = self._entity(["Apple", "AAPL", "Apple Inc."])
        assert _check_aliases(ent, "Apple") is True

    def test_case_insensitive_match(self) -> None:
        ent = self._entity(["apple inc."])
        assert _check_aliases(ent, "Apple Inc.") is True

    def test_no_match(self) -> None:
        ent = self._entity(["Microsoft", "MSFT"])
        assert _check_aliases(ent, "Apple") is False

    def test_none_aliases_returns_false(self) -> None:
        ent = self._entity(None)
        assert _check_aliases(ent, "Apple") is False

    def test_invalid_json_returns_false(self) -> None:
        ent = SimpleNamespace(aliases="not-valid-json[[[")
        assert _check_aliases(ent, "Apple") is False

    def test_empty_list_returns_false(self) -> None:
        ent = self._entity([])
        assert _check_aliases(ent, "Apple") is False


# ---------------------------------------------------------------------------
# Shared mock session builder
# ---------------------------------------------------------------------------

def _make_session(candidates: list[SimpleNamespace]) -> AsyncMock:
    """Return an AsyncSession mock whose execute() yields *candidates*."""
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = candidates
    session.execute.return_value = result
    return session


def _entity(
    name: str,
    canonical_name: str,
    entity_type: str = "ORG",
    aliases: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        name=name,
        canonical_name=canonical_name,
        type=entity_type,
        aliases=json.dumps(aliases) if aliases else None,
    )


# ---------------------------------------------------------------------------
# resolve_entity
# ---------------------------------------------------------------------------

class TestResolveEntityEmptyMention:
    @pytest.mark.asyncio
    async def test_empty_string_returns_none(self) -> None:
        session = _make_session([])
        assert await resolve_entity(session, "") is None

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_none(self) -> None:
        session = _make_session([])
        assert await resolve_entity(session, "   ") is None


class TestResolveEntityNoCandidates:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_candidates(self) -> None:
        session = _make_session([])
        result = await resolve_entity(session, "Unknown Corp")
        assert result is None


class TestResolveEntityTier1Exact:
    @pytest.mark.asyncio
    async def test_exact_canonical_name_match(self) -> None:
        ent = _entity("Apple Inc.", "Apple Inc.")
        session = _make_session([ent])
        result = await resolve_entity(session, "Apple Inc.")
        assert result is not None
        assert result.confidence == pytest.approx(1.0)
        assert result.method == "exact"
        assert result.entity_id == ent.id

    @pytest.mark.asyncio
    async def test_exact_match_case_insensitive(self) -> None:
        ent = _entity("apple inc.", "apple inc.")
        session = _make_session([ent])
        result = await resolve_entity(session, "Apple Inc.")
        assert result is not None
        assert result.method == "exact"

    @pytest.mark.asyncio
    async def test_exact_name_field_match(self) -> None:
        # canonical_name differs but name matches exactly.
        ent = _entity("Acme", "acme corp")
        session = _make_session([ent])
        result = await resolve_entity(session, "Acme")
        assert result is not None
        assert result.method == "exact"


class TestResolveEntityTier2Alias:
    @pytest.mark.asyncio
    async def test_alias_match_returns_confidence_095(self) -> None:
        ent = _entity("Apple Inc.", "apple inc.", aliases=["Apple", "AAPL"])
        session = _make_session([ent])
        # "Apple" is an alias but not an exact canonical/name match.
        result = await resolve_entity(session, "Apple")
        assert result is not None
        assert result.confidence == pytest.approx(0.95)
        assert result.method == "alias"

    @pytest.mark.asyncio
    async def test_alias_case_insensitive(self) -> None:
        ent = _entity("Apple Inc.", "apple inc.", aliases=["AAPL"])
        session = _make_session([ent])
        result = await resolve_entity(session, "aapl")
        assert result is not None
        assert result.method == "alias"


class TestResolveEntityTier3Fuzzy:
    @pytest.mark.asyncio
    async def test_fuzzy_match_above_threshold(self) -> None:
        # "Acme Corp" vs "Acme Corporation" — high similarity.
        ent = _entity("Acme Corporation", "Acme Corporation")
        session = _make_session([ent])
        result = await resolve_entity(session, "Acme Corp")
        assert result is not None
        assert result.method == "fuzzy"
        assert result.confidence >= _FUZZY_THRESHOLD

    @pytest.mark.asyncio
    async def test_fuzzy_below_threshold_returns_none(self) -> None:
        # Clearly unrelated strings.
        ent = _entity("Zeta Dynamics", "Zeta Dynamics")
        session = _make_session([ent])
        result = await resolve_entity(session, "Apple")
        assert result is None


# ---------------------------------------------------------------------------
# resolve_entity_filter
# ---------------------------------------------------------------------------

class TestResolveEntityFilter:
    @pytest.mark.asyncio
    async def test_none_filter_returns_none(self) -> None:
        session = _make_session([])
        result = await resolve_entity_filter(session, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_filter_returns_none(self) -> None:
        session = _make_session([])
        result = await resolve_entity_filter(session, "")
        assert result is None

    @pytest.mark.asyncio
    async def test_resolved_filter_returns_uuid(self) -> None:
        ent = _entity("Acme Corp", "acme corp")
        session = _make_session([ent])
        result = await resolve_entity_filter(session, "Acme Corp")
        assert result == ent.id

    @pytest.mark.asyncio
    async def test_unresolved_filter_returns_none(self) -> None:
        session = _make_session([])
        result = await resolve_entity_filter(session, "Completely Unknown Entity XYZ")
        assert result is None
