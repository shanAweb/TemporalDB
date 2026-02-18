from uuid import UUID

from pydantic import BaseModel, Field


class TextIngestRequest(BaseModel):
    """Request body for ingesting raw text."""
    text: str = Field(..., min_length=1, description="Raw text content to ingest")
    source: str = Field(..., min_length=1, description="Source identifier (e.g. 'quarterly-report')")
    metadata: dict | None = Field(default=None, description="Optional document metadata")


class IngestResponse(BaseModel):
    """Response after submitting a document for ingestion."""
    document_id: UUID
    source: str
    filename: str | None = None
    status: str = Field(default="processing", description="processing | completed | failed")
    message: str = "Document accepted for processing"
