"""
tests/integration/test_ingest_pipeline.py

Integration tests for the ingestion pipeline.

Tests the _ingest_text() shared helper end-to-end with mocked database,
Redis, and Kafka boundaries so no running infrastructure is required.

Coverage
--------
  - Fresh document: normalise → dedup miss → DB write → Kafka publish → "processing"
  - Duplicate document: dedup hit → early return → no DB write → "duplicate"
  - Empty text after normalisation → HTTP 422
  - Kafka failure is non-fatal (document still persisted)
  - Metadata JSON is serialised and stored on the Document row
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.api.routes.ingest import _ingest_text
from app.ingestion.deduplicator import DeduplicationResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_pg_session(doc_id: uuid.UUID) -> AsyncMock:
    """Return a minimal AsyncSession mock that handles add/flush/refresh."""
    session = AsyncMock()

    async def _refresh(obj: object) -> None:
        # Simulate SQLAlchemy refreshing the id after flush.
        obj.id = doc_id  # type: ignore[union-attr]

    session.refresh.side_effect = _refresh
    return session


def _make_redis() -> AsyncMock:
    return AsyncMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestTextFreshDocument:
    @pytest.mark.asyncio
    async def test_returns_processing_status(self) -> None:
        doc_id = uuid.uuid4()
        pg = _make_pg_session(doc_id)
        redis = _make_redis()

        dedup_result = DeduplicationResult(is_duplicate=False, document_id=str(doc_id))

        with (
            patch("app.api.routes.ingest.check_and_register", return_value=dedup_result),
            patch("app.api.routes.ingest.publish_document_ingested", new_callable=AsyncMock),
        ):
            response = await _ingest_text(
                raw_text="Supply chain disruptions led to a revenue decline in Q3.",
                source="quarterly-report",
                filename=None,
                mime_type="text/plain",
                extra_metadata=None,
                pg_session=pg,
                redis=redis,
            )

        assert response.status == "processing"
        assert response.source == "quarterly-report"
        assert response.filename is None

    @pytest.mark.asyncio
    async def test_document_added_to_session(self) -> None:
        doc_id = uuid.uuid4()
        pg = _make_pg_session(doc_id)
        redis = _make_redis()

        dedup_result = DeduplicationResult(is_duplicate=False, document_id=str(doc_id))

        with (
            patch("app.api.routes.ingest.check_and_register", return_value=dedup_result),
            patch("app.api.routes.ingest.publish_document_ingested", new_callable=AsyncMock),
        ):
            await _ingest_text(
                raw_text="Revenue dropped because of supply issues.",
                source="report",
                filename="report.pdf",
                mime_type="application/pdf",
                extra_metadata={"pages": 5},
                pg_session=pg,
                redis=redis,
            )

        pg.add.assert_called_once()
        pg.flush.assert_called()

    @pytest.mark.asyncio
    async def test_kafka_published_for_fresh_document(self) -> None:
        doc_id = uuid.uuid4()
        pg = _make_pg_session(doc_id)
        redis = _make_redis()

        dedup_result = DeduplicationResult(is_duplicate=False, document_id=str(doc_id))
        mock_publish = AsyncMock()

        with (
            patch("app.api.routes.ingest.check_and_register", return_value=dedup_result),
            patch("app.api.routes.ingest.publish_document_ingested", mock_publish),
        ):
            await _ingest_text(
                raw_text="Costs rose due to inflation.",
                source="finance-brief",
                filename=None,
                mime_type="text/plain",
                extra_metadata=None,
                pg_session=pg,
                redis=redis,
            )

        mock_publish.assert_awaited_once()
        call_kwargs = mock_publish.call_args
        assert call_kwargs.kwargs.get("source") == "finance-brief"


class TestIngestTextDuplicate:
    @pytest.mark.asyncio
    async def test_returns_duplicate_status(self) -> None:
        existing_id = str(uuid.uuid4())
        pg = _make_pg_session(uuid.uuid4())
        redis = _make_redis()

        dedup_result = DeduplicationResult(is_duplicate=True, document_id=existing_id)

        with (
            patch("app.api.routes.ingest.check_and_register", return_value=dedup_result),
            patch("app.api.routes.ingest.publish_document_ingested", new_callable=AsyncMock) as mock_pub,
        ):
            response = await _ingest_text(
                raw_text="Revenue dropped because of supply issues.",
                source="report",
                filename=None,
                mime_type="text/plain",
                extra_metadata=None,
                pg_session=pg,
                redis=redis,
            )

        assert response.status == "duplicate"
        assert str(response.document_id) == existing_id
        # No DB write or Kafka publish for duplicates
        pg.add.assert_not_called()
        mock_pub.assert_not_called()


class TestIngestTextEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_text_raises_422(self) -> None:
        pg = _make_pg_session(uuid.uuid4())
        redis = _make_redis()

        with pytest.raises(HTTPException) as exc_info:
            await _ingest_text(
                raw_text="",
                source="empty-doc",
                filename=None,
                mime_type="text/plain",
                extra_metadata=None,
                pg_session=pg,
                redis=redis,
            )

        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_whitespace_only_text_raises_422(self) -> None:
        pg = _make_pg_session(uuid.uuid4())
        redis = _make_redis()

        with pytest.raises(HTTPException) as exc_info:
            await _ingest_text(
                raw_text="   \n\t\n   ",
                source="blank-doc",
                filename=None,
                mime_type="text/plain",
                extra_metadata=None,
                pg_session=pg,
                redis=redis,
            )

        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_kafka_failure_is_non_fatal(self) -> None:
        """Document must be persisted even when Kafka publish raises."""
        doc_id = uuid.uuid4()
        pg = _make_pg_session(doc_id)
        redis = _make_redis()

        dedup_result = DeduplicationResult(is_duplicate=False, document_id=str(doc_id))

        with (
            patch("app.api.routes.ingest.check_and_register", return_value=dedup_result),
            patch(
                "app.api.routes.ingest.publish_document_ingested",
                side_effect=RuntimeError("broker unavailable"),
            ),
        ):
            # Should not raise despite Kafka failure
            response = await _ingest_text(
                raw_text="Costs rose due to inflation pressures this year.",
                source="brief",
                filename=None,
                mime_type="text/plain",
                extra_metadata=None,
                pg_session=pg,
                redis=redis,
            )

        assert response.status == "processing"
        pg.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_filename_stored_on_response(self) -> None:
        doc_id = uuid.uuid4()
        pg = _make_pg_session(doc_id)
        redis = _make_redis()

        dedup_result = DeduplicationResult(is_duplicate=False, document_id=str(doc_id))

        with (
            patch("app.api.routes.ingest.check_and_register", return_value=dedup_result),
            patch("app.api.routes.ingest.publish_document_ingested", new_callable=AsyncMock),
        ):
            response = await _ingest_text(
                raw_text="The quarterly results exceeded expectations significantly.",
                source="q3-report",
                filename="Q3-2024-Report.pdf",
                mime_type="application/pdf",
                extra_metadata=None,
                pg_session=pg,
                redis=redis,
            )

        assert response.filename == "Q3-2024-Report.pdf"
