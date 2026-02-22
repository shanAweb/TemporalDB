"""
tests/api/test_api_routes.py

HTTP-level smoke tests for the TemporalDB API routes.

All storage and messaging boundaries are mocked at the module level so
no running infrastructure is required.  The FastAPI dependency overrides
and lifespan patches are provided by tests/api/conftest.py.

Coverage
--------
  GET  /health              → 200, status key present
  POST /ingest              → 401 without key, 202 fresh, 202 duplicate
  POST /ingest              → 422 for empty text
  GET  /events              → 401 without key, 200 with paginated response
  GET  /events/{id}         → 200 when found, 404 when not found
  GET  /entities            → 401 without key, 200 with paginated response
  GET  /entities/{id}       → 200 when found, 404 when not found
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
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


def _mock_event() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        description="Costs rose due to inflation.",
        event_type="state_change",
        ts_start=datetime(2024, 7, 1, tzinfo=timezone.utc),
        ts_end=None,
        confidence=0.85,
        source_sentence="Costs rose due to inflation in Q3.",
        document_id=uuid.uuid4(),
        created_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )


def _mock_entity() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        name="Acme Corp",
        canonical_name="acme corp",
        type="ORG",
        description=None,
        aliases=None,
        created_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_body_contains_status(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_body_contains_app_name(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "app" in data

    def test_no_api_key_required(self, client: TestClient) -> None:
        # Health endpoint must be accessible without authentication.
        response = client.get("/health")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

class TestIngestTextRoute:
    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/ingest",
            json={"text": "Some content.", "source": "test"},
        )
        assert response.status_code == 401

    def test_wrong_api_key_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/ingest",
            json={"text": "Some content.", "source": "test"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_fresh_document_returns_202(self, client: TestClient) -> None:
        from app.ingestion.deduplicator import DeduplicationResult

        dedup = DeduplicationResult(is_duplicate=False, document_id=str(uuid.uuid4()))

        with (
            patch("app.api.routes.ingest.check_and_register", return_value=dedup),
            patch("app.api.routes.ingest.publish_document_ingested", new_callable=AsyncMock),
        ):
            response = client.post(
                "/ingest",
                json={"text": "Supply chain disruptions led to a revenue decline.", "source": "quarterly-report"},
                headers=_auth(),
            )

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "processing"
        assert data["source"] == "quarterly-report"

    def test_duplicate_document_returns_202_with_duplicate_status(
        self, client: TestClient
    ) -> None:
        from app.ingestion.deduplicator import DeduplicationResult

        existing_id = str(uuid.uuid4())
        dedup = DeduplicationResult(is_duplicate=True, document_id=existing_id)

        with (
            patch("app.api.routes.ingest.check_and_register", return_value=dedup),
            patch("app.api.routes.ingest.publish_document_ingested", new_callable=AsyncMock),
        ):
            response = client.post(
                "/ingest",
                json={"text": "Repeated document content.", "source": "source-a"},
                headers=_auth(),
            )

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "duplicate"
        assert data["document_id"] == existing_id

    def test_empty_text_returns_422(self, client: TestClient) -> None:
        # FastAPI validates min_length=1 on the request body field.
        response = client.post(
            "/ingest",
            json={"text": "", "source": "test"},
            headers=_auth(),
        )
        assert response.status_code == 422

    def test_missing_source_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/ingest",
            json={"text": "Some valid content."},
            headers=_auth(),
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /events
# ---------------------------------------------------------------------------

class TestListEventsRoute:
    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        response = client.get("/events")
        assert response.status_code == 401

    def test_returns_200_with_empty_list(self, client: TestClient) -> None:
        with patch(
            "app.api.routes.events.event_store.list_events",
            return_value=([], 0),
        ):
            response = client.get("/events", headers=_auth())

        assert response.status_code == 200
        data = response.json()
        assert data["events"] == []
        assert data["total"] == 0

    def test_returns_events_in_body(self, client: TestClient) -> None:
        ev = _mock_event()
        with patch(
            "app.api.routes.events.event_store.list_events",
            return_value=([ev], 1),
        ):
            response = client.get("/events", headers=_auth())

        assert response.status_code == 200
        data = response.json()
        assert len(data["events"]) == 1
        assert data["total"] == 1
        assert data["events"][0]["description"] == ev.description

    def test_pagination_fields_echoed(self, client: TestClient) -> None:
        with patch(
            "app.api.routes.events.event_store.list_events",
            return_value=([], 0),
        ):
            response = client.get("/events?offset=10&limit=5", headers=_auth())

        data = response.json()
        assert data["offset"] == 10
        assert data["limit"] == 5

    def test_limit_exceeds_max_returns_422(self, client: TestClient) -> None:
        response = client.get("/events?limit=999", headers=_auth())
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /events/{event_id}
# ---------------------------------------------------------------------------

class TestGetEventRoute:
    def test_found_event_returns_200(self, client: TestClient) -> None:
        ev = _mock_event()
        with patch(
            "app.api.routes.events.event_store.get_event_by_id",
            return_value=ev,
        ):
            response = client.get(f"/events/{ev.id}", headers=_auth())

        assert response.status_code == 200
        assert response.json()["id"] == str(ev.id)

    def test_missing_event_returns_404(self, client: TestClient) -> None:
        with patch(
            "app.api.routes.events.event_store.get_event_by_id",
            return_value=None,
        ):
            response = client.get(f"/events/{uuid.uuid4()}", headers=_auth())

        assert response.status_code == 404

    def test_invalid_uuid_returns_422(self, client: TestClient) -> None:
        response = client.get("/events/not-a-uuid", headers=_auth())
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /entities
# ---------------------------------------------------------------------------

class TestListEntitiesRoute:
    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        response = client.get("/entities")
        assert response.status_code == 401

    def test_returns_200_with_empty_list(self, client: TestClient) -> None:
        with patch(
            "app.api.routes.entities.entity_store.list_entities",
            return_value=([], 0),
        ):
            response = client.get("/entities", headers=_auth())

        assert response.status_code == 200
        data = response.json()
        assert data["entities"] == []
        assert data["total"] == 0

    def test_returns_entities_in_body(self, client: TestClient) -> None:
        ent = _mock_entity()
        with patch(
            "app.api.routes.entities.entity_store.list_entities",
            return_value=([ent], 1),
        ):
            response = client.get("/entities", headers=_auth())

        assert response.status_code == 200
        data = response.json()
        assert len(data["entities"]) == 1
        assert data["entities"][0]["name"] == ent.name


# ---------------------------------------------------------------------------
# GET /entities/{entity_id}
# ---------------------------------------------------------------------------

class TestGetEntityRoute:
    def test_found_entity_returns_200(self, client: TestClient) -> None:
        ent = _mock_entity()
        with patch(
            "app.api.routes.entities.entity_store.get_entity_by_id",
            return_value=ent,
        ):
            response = client.get(f"/entities/{ent.id}", headers=_auth())

        assert response.status_code == 200
        assert response.json()["name"] == ent.name

    def test_missing_entity_returns_404(self, client: TestClient) -> None:
        with patch(
            "app.api.routes.entities.entity_store.get_entity_by_id",
            return_value=None,
        ):
            response = client.get(f"/entities/{uuid.uuid4()}", headers=_auth())

        assert response.status_code == 404

    def test_invalid_uuid_returns_422(self, client: TestClient) -> None:
        response = client.get("/entities/not-a-uuid", headers=_auth())
        assert response.status_code == 422
