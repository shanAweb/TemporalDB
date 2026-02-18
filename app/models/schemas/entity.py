from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class EntityOut(BaseModel):
    """Response schema for an entity."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    canonical_name: str
    type: str
    description: str | None = None
    aliases: str | None = None
    created_at: datetime


class EntityBrief(BaseModel):
    """Compact entity representation for event responses."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    type: str


class EntityListResponse(BaseModel):
    """Paginated list of entities."""
    entities: list[EntityOut]
    total: int
    offset: int
    limit: int
