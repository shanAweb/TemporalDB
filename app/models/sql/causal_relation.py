import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.postgres import Base


class CausalRelation(Base):
    __tablename__ = "causal_relations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    cause_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    effect_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    confidence: Mapped[float] = mapped_column(
        Float, default=1.0, server_default=text("1.0"),
        comment="Confidence score for this causal link 0.0-1.0",
    )
    evidence: Mapped[str | None] = mapped_column(
        Text, comment="The phrase or sentence indicating this causal relationship"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
    )

    # Relationships
    cause_event: Mapped["Event"] = relationship(  # noqa: F821
        foreign_keys=[cause_event_id], back_populates="causes"
    )
    effect_event: Mapped["Event"] = relationship(  # noqa: F821
        foreign_keys=[effect_event_id], back_populates="caused_by"
    )

    def __repr__(self) -> str:
        return f"<CausalRelation(cause={self.cause_event_id}, effect={self.effect_event_id})>"
