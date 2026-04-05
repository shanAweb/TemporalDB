"""
tests/unit/test_entity_store.py

Unit tests for app.storage.entity_store.

All tests use AsyncMock SQLAlchemy / Neo4j sessions so no running
database is required.

Coverage
--------
  - _merge_aliases: adds new name, deduplicates, handles None input
  - upsert_entity: creates new entity (add+flush+refresh) when not found
  - upsert_entity: merges alias into existing entity when found
  - upsert_entity: calls graph_store when neo4j_session is provided
  - upsert_entity: skips graph_store when neo4j_session is None
  - get_entity_by_id: found → entity; not found → None
  - get_entity_by_canonical_name: found → entity; not found → None
  - list_entities: (results, total) from dual execute; empty set
  - delete_entity: found → True; not found → False; neo4j cypher on delete
  - bulk_upsert_entities: empty list returns []; same key reuses cache
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.storage.entity_store import (
    _merge_aliases,
    bulk_upsert_entities,
    delete_entity,
    get_entity_by_canonical_name,
    get_entity_by_id,
    list_entities,
    upsert_entity,
)


# ---------------------------------------------------------------------------
# Session / result mock helpers
# ---------------------------------------------------------------------------

def _session() -> AsyncMock:
    return AsyncMock()


def _scalar_one_or_none(value: object) -> MagicMock:
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _scalar_one(value: object) -> MagicMock:
    r = MagicMock()
    r.scalar_one.return_value = value
    return r


def _scalars_all(rows: list) -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.all.return_value = rows
    return r


def _mock_entity(
    canonical_name: str = "acme corp",
    name: str = "Acme Corp",
    entity_type: str = "ORG",
    aliases: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        name=name,
        canonical_name=canonical_name,
        type=entity_type,
        description=None,
        aliases=json.dumps(aliases or [name]),
        created_at=None,
    )


# ---------------------------------------------------------------------------
# _merge_aliases
# ---------------------------------------------------------------------------

class TestMergeAliases:
    def test_adds_new_name(self) -> None:
        result = _merge_aliases('["Apple Inc."]', "Apple")
        aliases = json.loads(result)
        assert "Apple" in aliases
        assert "Apple Inc." in aliases

    def test_deduplicates_existing_name(self) -> None:
        result = _merge_aliases('["Apple"]', "Apple")
        aliases = json.loads(result)
        assert aliases.count("Apple") == 1

    def test_none_input_creates_new_list(self) -> None:
        result = _merge_aliases(None, "Acme Corp")
        aliases = json.loads(result)
        assert aliases == ["Acme Corp"]

    def test_returns_json_string(self) -> None:
        result = _merge_aliases(None, "test")
        assert isinstance(result, str)
        assert json.loads(result) is not None


# ---------------------------------------------------------------------------
# upsert_entity
# ---------------------------------------------------------------------------

class TestUpsertEntityCreate:
    @pytest.mark.asyncio
    async def test_add_called_for_new_entity(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none(None)
        with patch("app.storage.entity_store.graph_store.upsert_entity_node"):
            await upsert_entity(session, name="Acme", canonical_name="Acme Corp", entity_type="ORG")
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_and_refresh_called_for_new_entity(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none(None)
        with patch("app.storage.entity_store.graph_store.upsert_entity_node"):
            await upsert_entity(session, name="Acme", canonical_name="Acme Corp", entity_type="ORG")
        session.flush.assert_awaited()
        session.refresh.assert_awaited()

    @pytest.mark.asyncio
    async def test_graph_store_called_when_neo4j_session_provided(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none(None)
        neo4j = AsyncMock()
        with patch("app.storage.entity_store.graph_store.upsert_entity_node") as mock_upsert:
            await upsert_entity(
                session,
                name="Acme",
                canonical_name="Acme Corp",
                entity_type="ORG",
                neo4j_session=neo4j,
            )
        mock_upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graph_store_skipped_when_no_neo4j_session(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none(None)
        with patch("app.storage.entity_store.graph_store.upsert_entity_node") as mock_upsert:
            await upsert_entity(
                session,
                name="Acme",
                canonical_name="Acme Corp",
                entity_type="ORG",
                neo4j_session=None,
            )
        mock_upsert.assert_not_awaited()


class TestUpsertEntityMerge:
    @pytest.mark.asyncio
    async def test_alias_merged_for_existing_entity(self) -> None:
        existing = _mock_entity(aliases=["Acme Corp"])
        session = _session()
        session.execute.return_value = _scalar_one_or_none(existing)
        with patch("app.storage.entity_store.graph_store.upsert_entity_node"):
            result = await upsert_entity(
                session, name="Acme", canonical_name="Acme Corp", entity_type="ORG"
            )
        aliases = json.loads(result.aliases)
        assert "Acme" in aliases

    @pytest.mark.asyncio
    async def test_add_not_called_for_existing_entity(self) -> None:
        existing = _mock_entity()
        session = _session()
        session.execute.return_value = _scalar_one_or_none(existing)
        with patch("app.storage.entity_store.graph_store.upsert_entity_node"):
            await upsert_entity(session, name="Acme", canonical_name="Acme Corp", entity_type="ORG")
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_called_for_existing_entity(self) -> None:
        existing = _mock_entity()
        session = _session()
        session.execute.return_value = _scalar_one_or_none(existing)
        with patch("app.storage.entity_store.graph_store.upsert_entity_node"):
            await upsert_entity(session, name="Acme", canonical_name="Acme Corp", entity_type="ORG")
        session.flush.assert_awaited()


# ---------------------------------------------------------------------------
# get_entity_by_id
# ---------------------------------------------------------------------------

class TestGetEntityById:
    @pytest.mark.asyncio
    async def test_found_returns_entity(self) -> None:
        ent = _mock_entity()
        session = _session()
        session.execute.return_value = _scalar_one_or_none(ent)
        result = await get_entity_by_id(session, ent.id)
        assert result is ent

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none(None)
        result = await get_entity_by_id(session, uuid.uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# get_entity_by_canonical_name
# ---------------------------------------------------------------------------

class TestGetEntityByCanonicalName:
    @pytest.mark.asyncio
    async def test_found_returns_entity(self) -> None:
        ent = _mock_entity()
        session = _session()
        session.execute.return_value = _scalar_one_or_none(ent)
        result = await get_entity_by_canonical_name(session, "Acme Corp", "ORG")
        assert result is ent

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none(None)
        result = await get_entity_by_canonical_name(session, "Unknown", "ORG")
        assert result is None


# ---------------------------------------------------------------------------
# list_entities
# ---------------------------------------------------------------------------

class TestListEntities:
    @pytest.mark.asyncio
    async def test_returns_entities_and_total(self) -> None:
        ent = _mock_entity()
        session = _session()
        session.execute.side_effect = [
            _scalar_one(1),
            _scalars_all([ent]),
        ]
        entities, total = await list_entities(session)
        assert total == 1
        assert entities == [ent]

    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        session = _session()
        session.execute.side_effect = [
            _scalar_one(0),
            _scalars_all([]),
        ]
        entities, total = await list_entities(session)
        assert entities == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_execute_called_twice(self) -> None:
        session = _session()
        session.execute.side_effect = [_scalar_one(0), _scalars_all([])]
        await list_entities(session)
        assert session.execute.await_count == 2


# ---------------------------------------------------------------------------
# delete_entity
# ---------------------------------------------------------------------------

class TestDeleteEntity:
    @pytest.mark.asyncio
    async def test_found_returns_true(self) -> None:
        entity_id = uuid.uuid4()
        session = _session()
        session.execute.return_value = _scalar_one_or_none(entity_id)
        result = await delete_entity(session, entity_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_not_found_returns_false(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none(None)
        result = await delete_entity(session, uuid.uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_neo4j_cypher_run_on_delete(self) -> None:
        entity_id = uuid.uuid4()
        session = _session()
        session.execute.return_value = _scalar_one_or_none(entity_id)
        neo4j = AsyncMock()
        await delete_entity(session, entity_id, neo4j_session=neo4j)
        neo4j.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_neo4j_not_called_when_not_found(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none(None)
        neo4j = AsyncMock()
        await delete_entity(session, uuid.uuid4(), neo4j_session=neo4j)
        neo4j.run.assert_not_awaited()


# ---------------------------------------------------------------------------
# bulk_upsert_entities
# ---------------------------------------------------------------------------

class TestBulkUpsertEntities:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self) -> None:
        session = _session()
        result = await bulk_upsert_entities(session, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_same_key_reuses_cache(self) -> None:
        """Two entities with the same (canonical_name, entity_type) → upsert called once."""
        ent = _mock_entity()
        session = _session()
        # First upsert_entity call finds nothing → creates entity.
        session.execute.return_value = _scalar_one_or_none(None)

        specs = [
            {"name": "Apple Inc.", "canonical_name": "apple inc.", "entity_type": "ORG"},
            {"name": "Apple",      "canonical_name": "apple inc.", "entity_type": "ORG"},
        ]
        with patch("app.storage.entity_store.graph_store.upsert_entity_node"):
            results = await bulk_upsert_entities(session, specs)

        # Two results returned for two specs.
        assert len(results) == 2
        # But both point to the same cached entity.
        assert results[0] is results[1]

    @pytest.mark.asyncio
    async def test_different_keys_upsert_separately(self) -> None:
        session = _session()
        session.execute.return_value = _scalar_one_or_none(None)

        specs = [
            {"name": "Apple Inc.", "canonical_name": "apple inc.", "entity_type": "ORG"},
            {"name": "John Smith", "canonical_name": "john smith", "entity_type": "PERSON"},
        ]
        with patch("app.storage.entity_store.graph_store.upsert_entity_node"):
            results = await bulk_upsert_entities(session, specs)

        assert len(results) == 2
        # Different keys → different objects.
        assert results[0] is not results[1]
