from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


ConnectorType = Literal["jira", "clickup", "timedoctor"]
SyncStatus = Literal["running", "success", "partial", "error"]


# ── Sync Run ──────────────────────────────────────────────────────────────────

class SyncRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    connector_id: UUID
    started_at: datetime
    finished_at: datetime | None = None
    status: SyncStatus
    items_fetched: int
    items_ingested: int
    items_skipped: int
    error_message: str | None = None


class SyncRunListResponse(BaseModel):
    runs: list[SyncRunOut]
    total: int
    offset: int
    limit: int


# ── Connector ─────────────────────────────────────────────────────────────────

class ConnectorCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    connector_type: ConnectorType
    credentials: dict[str, Any] = Field(
        ..., description="Plain-text credentials — encrypted server-side before storage",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-connector options (project keys, workspace IDs, etc.)",
    )
    sync_schedule: str = Field(
        default="0 * * * *",
        description="Cron expression, e.g. '0 * * * *' for hourly",
    )


class ConnectorUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    credentials: dict[str, Any] | None = Field(
        default=None, description="Re-encrypt with new credentials if provided",
    )
    config: dict[str, Any] | None = None
    sync_schedule: str | None = None
    is_enabled: bool | None = None


class ConnectorOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    connector_type: ConnectorType
    is_enabled: bool
    config: str | None = None
    sync_schedule: str
    last_synced_at: datetime | None = None
    last_sync_status: SyncStatus | None = None
    created_at: datetime
    updated_at: datetime
    last_run: SyncRunOut | None = None


class ConnectorListResponse(BaseModel):
    connectors: list[ConnectorOut]
    total: int
    offset: int
    limit: int


class SyncTriggerResponse(BaseModel):
    connector_id: UUID
    run_id: UUID
    message: str


class ValidateCredentialsResponse(BaseModel):
    valid: bool
    error: str | None = None
