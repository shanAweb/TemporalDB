"""add connector tables

Revision ID: a1b2c3d4e5f6
Revises: c73d4a0b0c3e
Create Date: 2026-04-04 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "c73d4a0b0c3e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── connectors ────────────────────────────────────────────────────────────
    op.create_table(
        "connectors",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.String(length=128),
            nullable=False,
            comment="Display label, e.g. 'Jira Production'",
        ),
        sa.Column(
            "connector_type",
            sa.String(length=32),
            nullable=False,
            comment="One of: jira, clickup, timedoctor",
        ),
        sa.Column(
            "is_enabled",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "credentials_enc",
            sa.Text(),
            nullable=True,
            comment="AES-256-GCM encrypted JSON blob of API credentials",
        ),
        sa.Column(
            "config",
            sa.Text(),
            nullable=True,
            comment="JSON-encoded connector config (project keys, workspace IDs, etc.)",
        ),
        sa.Column(
            "sync_schedule",
            sa.String(length=64),
            server_default="0 * * * *",
            nullable=False,
            comment="Cron expression controlling sync frequency",
        ),
        sa.Column(
            "last_synced_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp of the last completed sync attempt",
        ),
        sa.Column(
            "last_sync_status",
            sa.String(length=16),
            nullable=True,
            comment="One of: success, partial, error",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_connectors")),
    )
    op.create_index(
        op.f("ix_connectors_connector_type"), "connectors", ["connector_type"], unique=False
    )

    # ── connector_sync_runs ───────────────────────────────────────────────────
    op.create_table(
        "connector_sync_runs",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("connector_id", sa.UUID(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            comment="One of: running, success, partial, error",
        ),
        sa.Column(
            "items_fetched",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
            comment="Total items returned by the external API",
        ),
        sa.Column(
            "items_ingested",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
            comment="New documents created (deduped items excluded)",
        ),
        sa.Column(
            "items_skipped",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
            comment="Duplicate items skipped by deduplication",
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
            comment="Last error or truncated traceback",
        ),
        sa.Column(
            "metadata",
            sa.Text(),
            nullable=True,
            comment="JSON blob for incremental sync cursor and extra run info",
        ),
        sa.ForeignKeyConstraint(
            ["connector_id"],
            ["connectors.id"],
            name=op.f("fk_connector_sync_runs_connector_id_connectors"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_connector_sync_runs")),
    )
    op.create_index(
        op.f("ix_connector_sync_runs_connector_id"),
        "connector_sync_runs",
        ["connector_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_connector_sync_runs_connector_id"), table_name="connector_sync_runs"
    )
    op.drop_table("connector_sync_runs")
    op.drop_index(op.f("ix_connectors_connector_type"), table_name="connectors")
    op.drop_table("connectors")
