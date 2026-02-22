"""
app/api/routes/ingest.py

Ingestion endpoints.

POST /ingest
    Accept raw text + source identifier.  Normalise, deduplicate, persist
    to PostgreSQL, and publish a Kafka event for the NLP worker.

POST /ingest/file
    Accept a multipart file upload (PDF, DOCX, TXT, Markdown).  Extract
    text via FileConnector, then follow the same normalise → dedup →
    persist → publish pipeline.

Both endpoints return immediately with status "processing" — the NLP
pipeline runs asynchronously in the nlp_worker process.  Duplicate
submissions return status "duplicate" with the existing document_id.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import require_api_key
from app.database.postgres import get_db
from app.database.redis import get_redis
from app.ingestion.connectors.file import FileConnector
from app.ingestion.deduplicator import check_and_register
from app.ingestion.normalizer import normalize
from app.ingestion.producer import publish_document_ingested
from app.models.schemas.ingest import IngestResponse, TextIngestRequest
from app.models.sql.document import Document

logger = structlog.get_logger(__name__)

router = APIRouter()

_file_connector = FileConnector()

# Maximum allowed file size (50 MB)
_MAX_FILE_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Shared ingestion pipeline
# ---------------------------------------------------------------------------

async def _ingest_text(
    raw_text: str,
    source: str,
    filename: str | None,
    mime_type: str | None,
    extra_metadata: dict | None,
    pg_session: AsyncSession,
    redis: Redis,
) -> IngestResponse:
    """Core ingestion logic shared by both endpoints.

    Steps:
        1. Normalise text.
        2. Deduplicate via Redis fingerprint.
        3. Persist Document to PostgreSQL.
        4. Publish Kafka event.

    Args:
        raw_text:       Extracted document text.
        source:         Source identifier string.
        filename:       Original filename (None for raw-text ingestion).
        mime_type:      MIME type string (None for raw-text ingestion).
        extra_metadata: Optional metadata dict to store with the document.
        pg_session:     SQLAlchemy async session.
        redis:          Async Redis client.

    Returns:
        IngestResponse with status "processing" or "duplicate".
    """
    # ── 1. Normalise ──────────────────────────────────────────────────────
    normalised = normalize(raw_text)
    if not normalised:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document text is empty after normalisation.",
        )

    # ── 2. Deduplicate ────────────────────────────────────────────────────
    new_doc_id = str(uuid.uuid4())
    dedup = await check_and_register(normalised, new_doc_id, redis)

    if dedup.is_duplicate:
        logger.info(
            "ingest_duplicate",
            source=source,
            existing_id=dedup.document_id,
        )
        return IngestResponse(
            document_id=uuid.UUID(dedup.document_id),
            source=source,
            filename=filename,
            status="duplicate",
            message="Document already exists.",
        )

    # ── 3. Persist Document ───────────────────────────────────────────────
    metadata_json: str | None = None
    if extra_metadata:
        try:
            metadata_json = json.dumps(extra_metadata)
        except (TypeError, ValueError):
            metadata_json = None

    document = Document(
        id=uuid.UUID(new_doc_id),
        source=source,
        filename=filename,
        content_hash=None,   # fingerprint stored in Redis, not here
        raw_text=normalised,
        mime_type=mime_type,
        metadata_=metadata_json,
    )
    pg_session.add(document)
    await pg_session.flush()
    await pg_session.refresh(document)

    # ── 4. Publish Kafka event ────────────────────────────────────────────
    try:
        await publish_document_ingested(
            document_id=str(document.id),
            source=source,
            filename=filename,
        )
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: the document is saved; the worker can be triggered later.
        logger.warning("kafka_publish_failed", error=str(exc), document_id=str(document.id))

    logger.info(
        "ingest_accepted",
        document_id=str(document.id),
        source=source,
        filename=filename,
        text_length=len(normalised),
    )
    return IngestResponse(
        document_id=document.id,
        source=source,
        filename=filename,
        status="processing",
        message="Document accepted for processing.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest raw text",
)
async def ingest_text(
    body: TextIngestRequest,
    pg_session: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    _key: str = Depends(require_api_key),
) -> IngestResponse:
    """Ingest a raw text document.

    Normalises the text, deduplicates against previously ingested content,
    persists to PostgreSQL, and enqueues for NLP processing via Kafka.

    Returns immediately with ``status: processing``.  If the same content
    has already been ingested, returns ``status: duplicate`` with the
    existing ``document_id``.
    """
    return await _ingest_text(
        raw_text=body.text,
        source=body.source,
        filename=None,
        mime_type="text/plain",
        extra_metadata=body.metadata,
        pg_session=pg_session,
        redis=redis,
    )


@router.post(
    "/file",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a file upload",
)
async def ingest_file(
    file: UploadFile = File(..., description="PDF, DOCX, TXT, or Markdown file"),
    pg_session: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    _key: str = Depends(require_api_key),
) -> IngestResponse:
    """Ingest a file upload (PDF, DOCX, TXT, Markdown).

    Reads the uploaded file, extracts text via FileConnector, then follows
    the same normalise → dedup → persist → publish pipeline as the raw-text
    endpoint.

    Maximum file size: 50 MB.
    """
    # Size guard
    contents = await file.read()
    if len(contents) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {_MAX_FILE_BYTES // (1024*1024)} MB.",
        )

    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower() or ".bin"

    # Write to a temp file so FileConnector can work with a path.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = await _file_connector.extract(tmp_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    source = Path(filename).stem  # use filename stem as the source identifier

    return await _ingest_text(
        raw_text=result.text,
        source=source,
        filename=filename,
        mime_type=file.content_type,
        extra_metadata=result.metadata,
        pg_session=pg_session,
        redis=redis,
    )
