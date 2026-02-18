from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EventOut(BaseModel):
    """Response schema for an event."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    description: str
    event_type: str | None = None
    ts_start: datetime | None = None
    ts_end: datetime | None = None
    confidence: float
    source_sentence: str | None = None
    document_id: UUID
    created_at: datetime


class EventBrief(BaseModel):
    """Compact event representation for causal chains and lists."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    description: str
    ts_start: datetime | None = None
    confidence: float


class EventListResponse(BaseModel):
    """Paginated list of events."""
    events: list[EventOut]
    total: int
    offset: int
    limit: int
