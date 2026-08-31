"""Create managed model registry storage.

Revision ID: 0003_models
Revises: 0002_cameras
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_models"
down_revision: str | None = "0002_cameras"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("backend", sa.String(length=64), nullable=False),
        sa.Column("adapter", sa.String(length=128), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_json", sa.Text(), nullable=False),
        sa.Column("validation_state", sa.String(length=32), nullable=False),
        sa.Column("validation_details_json", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("source_license", sa.String(length=255), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("last_validated_at", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_models_active", "models", ["active"])
    op.create_index("ix_models_validation_state", "models", ["validation_state"])


def downgrade() -> None:
    op.drop_index("ix_models_validation_state", table_name="models")
    op.drop_index("ix_models_active", table_name="models")
    op.drop_table("models")
