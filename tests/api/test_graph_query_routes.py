"""
tests/api/test_graph_query_routes.py

HTTP-level smoke tests for the graph and query API routes.

GET  /graph/entity/{entity_id}
POST /query

All storage and orchestrator boundaries are mocked so no running
infrastructure is required.  The shared `client` fixture (conftest.py)
patches lifespan and overrides get_db, get_redis, and get_neo4j.

Coverage
--------
  GET  /graph/entity/{id}  → 401 without key
  GET  /graph/entity/{id}  → 404 when entity not in PostgreSQL
  GET  /graph/entity/{id}  → 200 with entity node + event node + edges
  GET  /graph/entity/{id}  → 200 with empty graph (no events)
  GET  /graph/entity/{id}  → max_events param validated (0 → 422)
  POST /query              → 401 without key
  POST /query              → 200 with synthesised answer
  POST /query              → 422 when question is missing
  POST /query              → response fields all present
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import VALID_API_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth(key: str = VALID_API_KEY) -> dict[str, str]:
    return {"X-API-Key": key}


def _mock_entity(canonical_name: str = "Acme Corp") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        name=canonical_name,
        canonical_name=canonical_name,
        type="ORG",
        description=None,
        aliases=None,
    )


def _graph_data(
    event_id: uuid.UUID | None = None,
    cause_id: uuid.UUID | None = None,
    effect_id: uuid.UUID | None = None,
) -> dict:
    events = []
    edges = []
    if event_id:
        events.append({
            "event_id": str(event_id),
            "description": "Costs rose due to inflation.",
            "event_type": "state_change",
            "ts_start": "2024-07-01T00:00:00",
            "confidence": 0.85,
        })
    if cause_id and effect_id:
        edges.append({
            "cause_id": str(cause_id),
            "effect_id": str(effect_id),
            "confidence": 0.80,
        })
    return {"events": events, "edges": edges}


# ---------------------------------------------------------------------------
# GET /graph/entity/{entity_id}
# ---------------------------------------------------------------------------

class TestGraphEntityRoute:
    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        response = client.get(f"/graph/entity/{uuid.uuid4()}")
        assert response.status_code == 401

    def test_wrong_api_key_returns_401(self, client: TestClient) -> None:
        response = client.get(
            f"/graph/entity/{uuid.uuid4()}",
            headers={"X-API-Key": "bad-key"},
        )
        assert response.status_code == 401

    def test_entity_not_found_returns_404(self, client: TestClient) -> None:
        with patch(
            "app.api.routes.graph.entity_store.get_entity_by_id",
            return_value=None,
        ):
            response = client.get(
                f"/graph/entity/{uuid.uuid4()}",
                headers=_auth(),
            )
        assert response.status_code == 404

    def test_empty_graph_returns_200(self, client: TestClient) -> None:
        ent = _mock_entity()
        with (
            patch("app.api.routes.graph.entity_store.get_entity_by_id",
                  return_value=ent),
            patch("app.api.routes.graph.graph_store.get_entity_graph",
                  return_value=_graph_data()),
        ):
            response = client.get(
                f"/graph/entity/{ent.id}",
                headers=_auth(),
            )
        assert response.status_code == 200
        data = response.json()
        # Only the anchor entity node; no event nodes.
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["type"] == "entity"
        assert data["edges"] == []

    def test_graph_with_event_node_and_involves_edge(self, client: TestClient) -> None:
        ent = _mock_entity()
        ev_id = uuid.uuid4()
        with (
            patch("app.api.routes.graph.entity_store.get_entity_by_id",
                  return_value=ent),
            patch("app.api.routes.graph.graph_store.get_entity_graph",
                  return_value=_graph_data(event_id=ev_id)),
        ):
            response = client.get(
                f"/graph/entity/{ent.id}",
                headers=_auth(),
            )
        assert response.status_code == 200
        data = response.json()
        # Entity node + event node.
        assert len(data["nodes"]) == 2
        node_types = {n["type"] for n in data["nodes"]}
        assert "entity" in node_types
        assert "event" in node_types
        # One INVOLVES edge.
        assert len(data["edges"]) == 1
        assert data["edges"][0]["type"] == "INVOLVES"

    def test_graph_with_causes_edge(self, client: TestClient) -> None:
        ent = _mock_entity()
        ev1_id = uuid.uuid4()
        ev2_id = uuid.uuid4()
        graph = _graph_data(event_id=ev1_id, cause_id=ev1_id, effect_id=ev2_id)
        with (
            patch("app.api.routes.graph.entity_store.get_entity_by_id",
                  return_value=ent),
            patch("app.api.routes.graph.graph_store.get_entity_graph",
                  return_value=graph),
        ):
            response = client.get(
                f"/graph/entity/{ent.id}",
                headers=_auth(),
            )
        data = response.json()
        edge_types = {e["type"] for e in data["edges"]}
        assert "CAUSES" in edge_types

    def test_entity_label_is_canonical_name(self, client: TestClient) -> None:
        ent = _mock_entity("Acme Corporation")
        with (
            patch("app.api.routes.graph.entity_store.get_entity_by_id",
                  return_value=ent),
            patch("app.api.routes.graph.graph_store.get_entity_graph",
                  return_value=_graph_data()),
        ):
            response = client.get(
                f"/graph/entity/{ent.id}",
                headers=_auth(),
            )
        entity_node = next(
            n for n in response.json()["nodes"] if n["type"] == "entity"
        )
        assert entity_node["label"] == "Acme Corporation"

    def test_invalid_uuid_returns_422(self, client: TestClient) -> None:
        response = client.get("/graph/entity/not-a-uuid", headers=_auth())
        assert response.status_code == 422

    def test_max_events_zero_returns_422(self, client: TestClient) -> None:
        response = client.get(
            f"/graph/entity/{uuid.uuid4()}?max_events=0",
            headers=_auth(),
        )
        assert response.status_code == 422

    def test_max_events_above_limit_returns_422(self, client: TestClient) -> None:
        response = client.get(
            f"/graph/entity/{uuid.uuid4()}?max_events=999",
            headers=_auth(),
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

def _query_response_payload(intent: str = "CAUSAL_WHY") -> dict:
    return {
        "answer": "Supply chain issues caused revenue to decline.",
        "confidence": 0.85,
        "intent": intent,
        "causal_chain": [],
        "events": [],
        "sources": [],
    }


class TestQueryRoute:
    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/query",
            json={"question": "Why did revenue drop?"},
        )
        assert response.status_code == 401

    def test_wrong_api_key_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/query",
            json={"question": "Why did revenue drop?"},
            headers={"X-API-Key": "wrong"},
        )
        assert response.status_code == 401

    def test_missing_question_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/query",
            json={},
            headers=_auth(),
        )
        assert response.status_code == 422

    def test_empty_question_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/query",
            json={"question": ""},
            headers=_auth(),
        )
        assert response.status_code == 422

    def test_valid_query_returns_200(self, client: TestClient) -> None:
        from app.models.schemas.query import QueryResponse
        payload = _query_response_payload()
        mock_response = QueryResponse(**payload)

        with patch(
            "app.api.routes.query.handle_query",
            return_value=mock_response,
        ):
            response = client.post(
                "/query",
                json={"question": "Why did revenue drop in Q3?"},
                headers=_auth(),
            )

        assert response.status_code == 200

    def test_response_contains_answer(self, client: TestClient) -> None:
        from app.models.schemas.query import QueryResponse
        payload = _query_response_payload()
        mock_response = QueryResponse(**payload)

        with patch(
            "app.api.routes.query.handle_query",
            return_value=mock_response,
        ):
            response = client.post(
                "/query",
                json={"question": "Why did costs rise?"},
                headers=_auth(),
            )

        data = response.json()
        assert data["answer"] == payload["answer"]
        assert data["intent"] == "CAUSAL_WHY"
        assert "confidence" in data
        assert "causal_chain" in data
        assert "events" in data
        assert "sources" in data

    def test_optional_filters_accepted(self, client: TestClient) -> None:
        from app.models.schemas.query import QueryResponse
        mock_response = QueryResponse(**_query_response_payload("ENTITY_TIMELINE"))

        with patch(
            "app.api.routes.query.handle_query",
            return_value=mock_response,
        ):
            response = client.post(
                "/query",
                json={
                    "question": "Show me everything about Acme Corp.",
                    "entity_filter": "Acme Corp",
                    "max_causal_hops": 5,
                },
                headers=_auth(),
            )

        assert response.status_code == 200
