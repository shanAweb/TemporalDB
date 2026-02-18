from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.schemas.event import EventBrief


class TimeRange(BaseModel):
    """Time range filter for queries."""
    start: datetime
    end: datetime


class QueryRequest(BaseModel):
    """Request body for natural language queries."""
    question: str = Field(..., min_length=1, description="Natural language question")
    entity_filter: str | None = Field(default=None, description="Filter results by entity name")
    time_range: TimeRange | None = Field(default=None, description="Filter results by time range")
    max_causal_hops: int = Field(default=3, ge=1, le=10, description="Max depth for causal chain traversal")


class CausalChainLink(BaseModel):
    """A single link in a causal chain."""
    id: UUID
    description: str
    ts_start: datetime | None = None
    confidence: float


class SourceReference(BaseModel):
    """Reference to a source document."""
    id: UUID
    source: str
    metadata: dict | None = None


class QueryResponse(BaseModel):
    """Response to a natural language query."""
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    intent: str = Field(..., description="Classified intent: CAUSAL_WHY, TEMPORAL_RANGE, SIMILARITY, ENTITY_TIMELINE")
    causal_chain: list[CausalChainLink] = Field(default_factory=list)
    events: list[EventBrief] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
