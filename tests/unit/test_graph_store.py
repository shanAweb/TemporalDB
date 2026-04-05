"""
tests/unit/test_graph_store.py

Unit tests for app.storage.graph_store.

The Neo4j AsyncSession is fully mocked — no running Neo4j is required.

Coverage
--------
  _str: UUID→str, None→None
  _iso: datetime→ISO-8601 string, None→None
  upsert_event_node: session.run called; UUID params converted to str;
                     optional fields default to None; ts_start converted
  upsert_entity_node: session.run called; id and type params correct
  upsert_causal_edge: session.run called; cause/effect/relation_id as str;
                      confidence and evidence forwarded
  upsert_involves_edge: session.run called; event_id/entity_id as str
  delete_event_node: True when record deleted>0, False when 0,
                     False when result.single() returns None
  delete_causal_edge: True when record deleted>0, False when 0,
                      False when no record; relation_id as str
  get_causal_chain: returns result.data(); downstream/upstream/both clauses
                    built correctly; max_hops clamped to [1, 10];
                    seed id passed as str
  get_entity_graph: empty events → immediate {"events":[], "edges":[]} with
                    only one query issued; events present → two queries;
                    events + edges returned; entity_id as str; limit passed
  get_causal_path_between: max_hops clamped [1, 10]; source/target as str;
                            returns result.data()
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.storage.graph_store import (
    _iso,
    _str,
    delete_causal_edge,
    delete_event_node,
    get_causal_chain,
    get_causal_path_between,
    get_entity_graph,
    upsert_causal_edge,
    upsert_entity_node,
    upsert_event_node,
    upsert_involves_edge,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _run_result(
    data: list[dict] | None = None,
    single: dict | None = None,
) -> AsyncMock:
    """Return a mock Neo4j result whose .data() and .single() are pre-set."""
    result = AsyncMock()
    result.data   = AsyncMock(return_value=data or [])
    result.single = AsyncMock(return_value=single)
    return result


def _session(*results: AsyncMock) -> AsyncMock:
    """
    Return a mock AsyncSession.

    If one result is given, session.run always returns it.
    If multiple results are given, session.run returns them in sequence
    (first call → results[0], second call → results[1], …).
    """
    session = AsyncMock()
    if len(results) == 1:
        session.run = AsyncMock(return_value=results[0])
    else:
        session.run = AsyncMock(side_effect=list(results))
    return session


# ---------------------------------------------------------------------------
# Tests: _str helper
# ---------------------------------------------------------------------------

class TestStrHelper:
    def test_uuid_converted_to_string(self) -> None:
        uid = uuid.uuid4()
        assert _str(uid) == str(uid)

    def test_none_returns_none(self) -> None:
        assert _str(None) is None

    def test_string_is_lowercase_hyphenated(self) -> None:
        uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert _str(uid) == "12345678-1234-5678-1234-567812345678"


# ---------------------------------------------------------------------------
# Tests: _iso helper
# ---------------------------------------------------------------------------

class TestIsoHelper:
    def test_datetime_converted_to_isoformat(self) -> None:
        dt = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert _iso(dt) == dt.isoformat()

    def test_none_returns_none(self) -> None:
        assert _iso(None) is None

    def test_returns_str_type(self) -> None:
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert isinstance(_iso(dt), str)


# ---------------------------------------------------------------------------
# Tests: upsert_event_node
# ---------------------------------------------------------------------------

class TestUpsertEventNode:
    @pytest.mark.asyncio
    async def test_session_run_called_once(self) -> None:
        sess = _session(_run_result())
        await upsert_event_node(sess, event_id=uuid.uuid4(), description="Test")
        sess.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_event_id_passed_as_string(self) -> None:
        event_id = uuid.uuid4()
        sess = _session(_run_result())
        await upsert_event_node(sess, event_id=event_id, description="Test")
        params = sess.run.call_args[0][1]
        assert params["id"] == str(event_id)

    @pytest.mark.asyncio
    async def test_description_passed(self) -> None:
        sess = _session(_run_result())
        await upsert_event_node(sess, event_id=uuid.uuid4(), description="Revenue fell")
        params = sess.run.call_args[0][1]
        assert params["description"] == "Revenue fell"

    @pytest.mark.asyncio
    async def test_document_id_converted_to_string(self) -> None:
        doc_id = uuid.uuid4()
        sess = _session(_run_result())
        await upsert_event_node(
            sess, event_id=uuid.uuid4(), description="X", document_id=doc_id
        )
        params = sess.run.call_args[0][1]
        assert params["document_id"] == str(doc_id)

    @pytest.mark.asyncio
    async def test_ts_start_converted_to_iso(self) -> None:
        ts = datetime(2024, 7, 1, tzinfo=timezone.utc)
        sess = _session(_run_result())
        await upsert_event_node(sess, event_id=uuid.uuid4(), description="X", ts_start=ts)
        params = sess.run.call_args[0][1]
        assert params["ts_start"] == ts.isoformat()

    @pytest.mark.asyncio
    async def test_optional_fields_default_to_none(self) -> None:
        sess = _session(_run_result())
        await upsert_event_node(sess, event_id=uuid.uuid4(), description="X")
        params = sess.run.call_args[0][1]
        assert params["event_type"]      is None
        assert params["ts_start"]        is None
        assert params["ts_end"]          is None
        assert params["source_sentence"] is None
        assert params["document_id"]     is None


# ---------------------------------------------------------------------------
# Tests: upsert_entity_node
# ---------------------------------------------------------------------------

class TestUpsertEntityNode:
    @pytest.mark.asyncio
    async def test_session_run_called_once(self) -> None:
        sess = _session(_run_result())
        await upsert_entity_node(
            sess,
            entity_id=uuid.uuid4(),
            name="Acme",
            canonical_name="acme",
            entity_type="ORG",
        )
        sess.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_entity_id_passed_as_string(self) -> None:
        entity_id = uuid.uuid4()
        sess = _session(_run_result())
        await upsert_entity_node(
            sess,
            entity_id=entity_id,
            name="Acme",
            canonical_name="acme",
            entity_type="ORG",
        )
        params = sess.run.call_args[0][1]
        assert params["id"] == str(entity_id)

    @pytest.mark.asyncio
    async def test_type_param_forwarded(self) -> None:
        sess = _session(_run_result())
        await upsert_entity_node(
            sess,
            entity_id=uuid.uuid4(),
            name="Alice",
            canonical_name="alice",
            entity_type="PERSON",
        )
        params = sess.run.call_args[0][1]
        assert params["type"] == "PERSON"

    @pytest.mark.asyncio
    async def test_canonical_name_forwarded(self) -> None:
        sess = _session(_run_result())
        await upsert_entity_node(
            sess,
            entity_id=uuid.uuid4(),
            name="Acme Corp",
            canonical_name="acme corp",
            entity_type="ORG",
        )
        params = sess.run.call_args[0][1]
        assert params["canonical_name"] == "acme corp"


# ---------------------------------------------------------------------------
# Tests: upsert_causal_edge
# ---------------------------------------------------------------------------

class TestUpsertCausalEdge:
    @pytest.mark.asyncio
    async def test_session_run_called_once(self) -> None:
        sess = _session(_run_result())
        await upsert_causal_edge(
            sess,
            cause_event_id=uuid.uuid4(),
            effect_event_id=uuid.uuid4(),
            relation_id=uuid.uuid4(),
        )
        sess.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ids_converted_to_strings(self) -> None:
        cause_id    = uuid.uuid4()
        effect_id   = uuid.uuid4()
        relation_id = uuid.uuid4()
        sess = _session(_run_result())
        await upsert_causal_edge(
            sess,
            cause_event_id=cause_id,
            effect_event_id=effect_id,
            relation_id=relation_id,
        )
        params = sess.run.call_args[0][1]
        assert params["cause_id"]    == str(cause_id)
        assert params["effect_id"]   == str(effect_id)
        assert params["relation_id"] == str(relation_id)

    @pytest.mark.asyncio
    async def test_confidence_and_evidence_forwarded(self) -> None:
        sess = _session(_run_result())
        await upsert_causal_edge(
            sess,
            cause_event_id=uuid.uuid4(),
            effect_event_id=uuid.uuid4(),
            relation_id=uuid.uuid4(),
            confidence=0.75,
            evidence="due to inflation",
        )
        params = sess.run.call_args[0][1]
        assert params["confidence"] == pytest.approx(0.75)
        assert params["evidence"]   == "due to inflation"

    @pytest.mark.asyncio
    async def test_evidence_defaults_to_none(self) -> None:
        sess = _session(_run_result())
        await upsert_causal_edge(
            sess,
            cause_event_id=uuid.uuid4(),
            effect_event_id=uuid.uuid4(),
            relation_id=uuid.uuid4(),
        )
        params = sess.run.call_args[0][1]
        assert params["evidence"] is None


# ---------------------------------------------------------------------------
# Tests: upsert_involves_edge
# ---------------------------------------------------------------------------

class TestUpsertInvolvesEdge:
    @pytest.mark.asyncio
    async def test_session_run_called_once(self) -> None:
        sess = _session(_run_result())
        await upsert_involves_edge(sess, event_id=uuid.uuid4(), entity_id=uuid.uuid4())
        sess.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ids_converted_to_strings(self) -> None:
        event_id  = uuid.uuid4()
        entity_id = uuid.uuid4()
        sess = _session(_run_result())
        await upsert_involves_edge(sess, event_id=event_id, entity_id=entity_id)
        params = sess.run.call_args[0][1]
        assert params["event_id"]  == str(event_id)
        assert params["entity_id"] == str(entity_id)


# ---------------------------------------------------------------------------
# Tests: delete_event_node
# ---------------------------------------------------------------------------

class TestDeleteEventNode:
    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self) -> None:
        sess = _session(_run_result(single={"deleted": 1}))
        assert await delete_event_node(sess, uuid.uuid4()) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_count_zero(self) -> None:
        sess = _session(_run_result(single={"deleted": 0}))
        assert await delete_event_node(sess, uuid.uuid4()) is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_record(self) -> None:
        sess = _session(_run_result(single=None))
        assert await delete_event_node(sess, uuid.uuid4()) is False

    @pytest.mark.asyncio
    async def test_event_id_passed_as_string(self) -> None:
        event_id = uuid.uuid4()
        sess = _session(_run_result(single={"deleted": 0}))
        await delete_event_node(sess, event_id)
        params = sess.run.call_args[0][1]
        assert params["id"] == str(event_id)


# ---------------------------------------------------------------------------
# Tests: delete_causal_edge
# ---------------------------------------------------------------------------

class TestDeleteCausalEdge:
    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self) -> None:
        sess = _session(_run_result(single={"deleted": 1}))
        assert await delete_causal_edge(sess, uuid.uuid4()) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_count_zero(self) -> None:
        sess = _session(_run_result(single={"deleted": 0}))
        assert await delete_causal_edge(sess, uuid.uuid4()) is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_record(self) -> None:
        sess = _session(_run_result(single=None))
        assert await delete_causal_edge(sess, uuid.uuid4()) is False

    @pytest.mark.asyncio
    async def test_relation_id_passed_as_string(self) -> None:
        relation_id = uuid.uuid4()
        sess = _session(_run_result(single={"deleted": 0}))
        await delete_causal_edge(sess, relation_id)
        params = sess.run.call_args[0][1]
        assert params["relation_id"] == str(relation_id)


# ---------------------------------------------------------------------------
# Tests: get_causal_chain
# ---------------------------------------------------------------------------

class TestGetCausalChain:
    @pytest.mark.asyncio
    async def test_returns_records_from_result(self) -> None:
        rows = [{"event_id": str(uuid.uuid4()), "description": "X", "hop": 1}]
        sess = _session(_run_result(data=rows))
        assert await get_causal_chain(sess, uuid.uuid4()) == rows

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_list(self) -> None:
        sess = _session(_run_result(data=[]))
        assert await get_causal_chain(sess, uuid.uuid4()) == []

    @pytest.mark.asyncio
    async def test_downstream_uses_forward_arrow(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_causal_chain(sess, uuid.uuid4(), direction="downstream")
        cypher = sess.run.call_args[0][0]
        # (seed)-[:CAUSES*1..N]->(related)
        assert ")-[:CAUSES" in cypher
        assert "]->(related)" in cypher

    @pytest.mark.asyncio
    async def test_upstream_uses_backward_arrow(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_causal_chain(sess, uuid.uuid4(), direction="upstream")
        cypher = sess.run.call_args[0][0]
        # (seed)<-[:CAUSES*1..N]-(related)
        assert ")<-[:CAUSES" in cypher

    @pytest.mark.asyncio
    async def test_both_uses_undirected_pattern(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_causal_chain(sess, uuid.uuid4(), direction="both")
        cypher = sess.run.call_args[0][0]
        # (seed)-[:CAUSES*1..N]-(related)  — no right-arrow
        assert ")-[:CAUSES" in cypher
        assert "]->(related)" not in cypher

    @pytest.mark.asyncio
    async def test_max_hops_clamped_to_minimum_one(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_causal_chain(sess, uuid.uuid4(), max_hops=0)
        cypher = sess.run.call_args[0][0]
        assert "*1..1" in cypher

    @pytest.mark.asyncio
    async def test_max_hops_clamped_to_maximum_ten(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_causal_chain(sess, uuid.uuid4(), max_hops=99)
        cypher = sess.run.call_args[0][0]
        assert "*1..10" in cypher

    @pytest.mark.asyncio
    async def test_max_hops_within_range_respected(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_causal_chain(sess, uuid.uuid4(), max_hops=4)
        cypher = sess.run.call_args[0][0]
        assert "*1..4" in cypher

    @pytest.mark.asyncio
    async def test_seed_id_passed_as_string(self) -> None:
        event_id = uuid.uuid4()
        sess = _session(_run_result(data=[]))
        await get_causal_chain(sess, event_id)
        params = sess.run.call_args[0][1]
        assert params["id"] == str(event_id)


# ---------------------------------------------------------------------------
# Tests: get_entity_graph
# ---------------------------------------------------------------------------

class TestGetEntityGraph:
    @pytest.mark.asyncio
    async def test_empty_events_returns_empty_dict(self) -> None:
        sess = _session(_run_result(data=[]))
        result = await get_entity_graph(sess, uuid.uuid4())
        assert result == {"events": [], "edges": []}

    @pytest.mark.asyncio
    async def test_only_one_query_when_no_events(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_entity_graph(sess, uuid.uuid4())
        assert sess.run.await_count == 1

    @pytest.mark.asyncio
    async def test_two_queries_when_events_found(self) -> None:
        events = [{"event_id": str(uuid.uuid4()), "description": "X"}]
        sess = _session(
            _run_result(data=events),
            _run_result(data=[]),
        )
        await get_entity_graph(sess, uuid.uuid4())
        assert sess.run.await_count == 2

    @pytest.mark.asyncio
    async def test_events_and_edges_returned(self) -> None:
        event_id = str(uuid.uuid4())
        events = [{"event_id": event_id, "description": "Revenue fell"}]
        edges  = [{"cause_id": event_id, "effect_id": str(uuid.uuid4()), "confidence": 0.9}]
        sess = _session(
            _run_result(data=events),
            _run_result(data=edges),
        )
        result = await get_entity_graph(sess, uuid.uuid4())
        assert result["events"] == events
        assert result["edges"]  == edges

    @pytest.mark.asyncio
    async def test_entity_id_passed_as_string(self) -> None:
        entity_id = uuid.uuid4()
        sess = _session(_run_result(data=[]))
        await get_entity_graph(sess, entity_id)
        params = sess.run.call_args[0][1]
        assert params["entity_id"] == str(entity_id)

    @pytest.mark.asyncio
    async def test_max_events_limit_forwarded(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_entity_graph(sess, uuid.uuid4(), max_events=25)
        params = sess.run.call_args[0][1]
        assert params["limit"] == 25

    @pytest.mark.asyncio
    async def test_default_max_events_is_50(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_entity_graph(sess, uuid.uuid4())
        params = sess.run.call_args[0][1]
        assert params["limit"] == 50


# ---------------------------------------------------------------------------
# Tests: get_causal_path_between
# ---------------------------------------------------------------------------

class TestGetCausalPathBetween:
    @pytest.mark.asyncio
    async def test_returns_records_from_result(self) -> None:
        rows = [{"event_id": str(uuid.uuid4()), "description": "X"}]
        sess = _session(_run_result(data=rows))
        assert await get_causal_path_between(sess, uuid.uuid4(), uuid.uuid4()) == rows

    @pytest.mark.asyncio
    async def test_empty_when_no_path(self) -> None:
        sess = _session(_run_result(data=[]))
        assert await get_causal_path_between(sess, uuid.uuid4(), uuid.uuid4()) == []

    @pytest.mark.asyncio
    async def test_max_hops_clamped_to_minimum_one(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_causal_path_between(sess, uuid.uuid4(), uuid.uuid4(), max_hops=0)
        cypher = sess.run.call_args[0][0]
        assert "*1..1" in cypher

    @pytest.mark.asyncio
    async def test_max_hops_clamped_to_maximum_ten(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_causal_path_between(sess, uuid.uuid4(), uuid.uuid4(), max_hops=50)
        cypher = sess.run.call_args[0][0]
        assert "*1..10" in cypher

    @pytest.mark.asyncio
    async def test_max_hops_within_range_respected(self) -> None:
        sess = _session(_run_result(data=[]))
        await get_causal_path_between(sess, uuid.uuid4(), uuid.uuid4(), max_hops=3)
        cypher = sess.run.call_args[0][0]
        assert "*1..3" in cypher

    @pytest.mark.asyncio
    async def test_source_and_target_ids_as_strings(self) -> None:
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        sess = _session(_run_result(data=[]))
        await get_causal_path_between(sess, source_id, target_id)
        params = sess.run.call_args[0][1]
        assert params["source_id"] == str(source_id)
        assert params["target_id"] == str(target_id)
