import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.postgres import Base


class Connector(Base):
    __tablename__ = "connectors"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(
        String(128), nullable=False,
        comment="Display label, e.g. 'Jira Production'",
    )
    connector_type: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True,
        comment="One of: jira, clickup, timedoctor",
    )
    is_enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=text("true"), nullable=False,
    )
    credentials_enc: Mapped[str | None] = mapped_column(
        Text, comment="AES-256-GCM encrypted JSON blob of API credentials",
    )
    config: Mapped[str | None] = mapped_column(
        Text, comment="JSON-encoded connector config (project keys, workspace IDs, etc.)",
    )
    sync_schedule: Mapped[str] = mapped_column(
        String(64), nullable=False, default="0 * * * *",
        server_default="0 * * * *",
        comment="Cron expression controlling sync frequency",
    )
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="Timestamp of the last completed sync attempt",
    )
    last_sync_status: Mapped[str | None] = mapped_column(
        String(16),
        comment="One of: success, partial, error",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    sync_runs: Mapped[list["ConnectorSyncRun"]] = relationship(
        back_populates="connector", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Connector(id={self.id}, type='{self.connector_type}', name='{self.name}')>"


class ConnectorSyncRun(Base):
    __tablename__ = "connector_sync_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    connector_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("connectors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="running",
        comment="One of: running, success, partial, error",
    )
    items_fetched: Mapped[int] = mapped_column(
        Integer, default=0, server_default=text("0"), nullable=False,
        comment="Total items returned by the external API",
    )
    items_ingested: Mapped[int] = mapped_column(
        Integer, default=0, server_default=text("0"), nullable=False,
        comment="New documents created (deduped items excluded)",
    )
    items_skipped: Mapped[int] = mapped_column(
        Integer, default=0, server_default=text("0"), nullable=False,
        comment="Duplicate items skipped by deduplication",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, comment="Last error or truncated traceback",
    )
    metadata_: Mapped[str | None] = mapped_column(
        "metadata", type_=Text,
        comment="JSON blob for incremental sync cursor and extra run info",
    )

    # Relationships
    connector: Mapped["Connector"] = relationship(back_populates="sync_runs")

    def __repr__(self) -> str:
        return f"<ConnectorSyncRun(id={self.id}, connector_id={self.connector_id}, status='{self.status}')>"
