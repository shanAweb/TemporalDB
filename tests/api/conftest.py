"""
tests/api/conftest.py

Shared fixtures for API route tests.

The `client` fixture:
  - Patches all startup/shutdown infrastructure calls (Postgres, Neo4j,
    Redis, Kafka) so no running services are required.
  - Overrides the get_db and get_redis FastAPI dependencies with async
    generator mocks.
  - Leaves require_api_key using the real implementation; tests that need
    an authenticated client send the default "changeme" key (matches
    settings.api_key default).
  - Clears dependency_overrides after each test to avoid cross-test leakage.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.database.neo4j import get_neo4j
from app.database.postgres import get_db
from app.database.redis import get_redis
from app.main import app

# These are the coroutines called during app lifespan that would otherwise
# try to connect to real infrastructure.
_LIFESPAN_PATCHES = [
    "app.main.init_postgres",
    "app.main.init_neo4j",
    "app.main.init_redis",
    "app.main.init_kafka_producer",
    "app.main.close_postgres",
    "app.main.close_neo4j",
    "app.main.close_redis",
    "app.main.close_kafka_producer",
]

# Default API key that matches settings.api_key default value.
VALID_API_KEY = "changeme"


async def _mock_get_db() -> AsyncMock:  # type: ignore[override]
    """Async generator override for the get_db dependency."""
    yield AsyncMock()


async def _mock_get_redis() -> AsyncMock:  # type: ignore[override]
    """Async generator override for the get_redis dependency."""
    yield AsyncMock()


async def _mock_get_neo4j() -> AsyncMock:  # type: ignore[override]
    """Async generator override for the get_neo4j dependency."""
    yield AsyncMock()


@pytest.fixture()
def client() -> TestClient:  # type: ignore[return]
    """
    Return a TestClient with all infrastructure dependencies mocked.

    Yields inside a context manager so lifespan patches are active for the
    full duration of each test, and dependency_overrides are cleared on exit.
    """
    app.dependency_overrides[get_db] = _mock_get_db
    app.dependency_overrides[get_redis] = _mock_get_redis
    app.dependency_overrides[get_neo4j] = _mock_get_neo4j

    with (
        patch("app.main.init_postgres"),
        patch("app.main.init_neo4j"),
        patch("app.main.init_redis"),
        patch("app.main.init_kafka_producer"),
        patch("app.main.close_postgres"),
        patch("app.main.close_neo4j"),
        patch("app.main.close_redis"),
        patch("app.main.close_kafka_producer"),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    app.dependency_overrides.clear()
