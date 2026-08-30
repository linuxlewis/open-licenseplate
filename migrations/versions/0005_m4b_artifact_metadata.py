"""Add M4-B artifact evidence and explicit ranking.

Revision ID: 0005_m4b_artifact_metadata
Revises: 0004_events_artifacts
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_m4b_artifact_metadata"
down_revision: str | None = "0004_events_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "event_artifacts",
        sa.Column("artifact_rank", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "event_artifacts",
        sa.Column(
            "quality_evidence_json",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("event_artifacts", "quality_evidence_json")
    op.drop_column("event_artifacts", "artifact_rank")
