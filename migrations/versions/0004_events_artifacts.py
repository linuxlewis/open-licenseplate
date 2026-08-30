"""Create capture sessions, detection events, and event artifacts.

Revision ID: 0004_events_artifacts
Revises: 0003_models
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_events_artifacts"
down_revision: str | None = "0003_models"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "capture_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("camera_id", sa.String(length=36), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("model_checksum", sa.String(length=64), nullable=False),
        sa.Column("compute_configuration_json", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("ended_at", sa.Text(), nullable=True),
        sa.Column("end_reason", sa.String(length=64), nullable=True),
        sa.Column("negotiated_codec", sa.String(length=64), nullable=True),
        sa.Column("negotiated_width", sa.Integer(), nullable=True),
        sa.Column("negotiated_height", sa.Integer(), nullable=True),
        sa.Column("negotiated_fps", sa.Float(), nullable=True),
        sa.Column("application_version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["cameras.id"],
            name="fk_capture_sessions_camera_id_cameras",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["models.id"],
            name="fk_capture_sessions_model_id_models",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "detection_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("camera_id", sa.String(length=36), nullable=False),
        sa.Column("capture_session_id", sa.String(length=36), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("model_checksum", sa.String(length=64), nullable=False),
        sa.Column("first_seen_at", sa.Text(), nullable=False),
        sa.Column("last_seen_at", sa.Text(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column("maximum_confidence", sa.Float(), nullable=False),
        sa.Column("event_state", sa.String(length=32), nullable=False),
        sa.Column("best_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("crop_ranking_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["cameras.id"],
            name="fk_detection_events_camera_id_cameras",
        ),
        sa.ForeignKeyConstraint(
            ["capture_session_id"],
            ["capture_sessions.id"],
            name="fk_detection_events_capture_session_id_capture_sessions",
        ),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["models.id"],
            name="fk_detection_events_model_id_models",
        ),
        sa.ForeignKeyConstraint(
            ["best_artifact_id"],
            ["event_artifacts.id"],
            name="fk_detection_events_best_artifact_id_event_artifacts",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "capture_session_id",
            "track_id",
            name="uq_detection_events_capture_session_track",
        ),
    )

    op.create_table(
        "event_artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("artifact_kind", sa.String(length=64), nullable=False),
        sa.Column("managed_relative_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("source_frame_sequence", sa.Integer(), nullable=False),
        sa.Column("source_timestamp", sa.Text(), nullable=False),
        sa.Column("detection_confidence", sa.Float(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("quality_scoring_version", sa.String(length=64), nullable=False),
        sa.Column(
            "quality_evidence_json",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("deleted_at", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["detection_events.id"],
            name="fk_event_artifacts_event_id_detection_events",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "managed_relative_path",
            name="uq_event_artifacts_managed_relative_path",
        ),
    )

    op.create_index(
        "ix_capture_sessions_camera_started_at",
        "capture_sessions",
        ["camera_id", "started_at"],
    )
    op.create_index(
        "ix_capture_sessions_model_started_at",
        "capture_sessions",
        ["model_id", "started_at"],
    )
    op.create_index(
        "ix_detection_events_camera_first_seen_at",
        "detection_events",
        ["camera_id", "first_seen_at"],
    )
    op.create_index(
        "ix_detection_events_capture_session",
        "detection_events",
        ["capture_session_id"],
    )
    op.create_index(
        "ix_detection_events_event_state",
        "detection_events",
        ["event_state"],
    )
    op.create_index(
        "ix_detection_events_first_seen_at",
        "detection_events",
        ["first_seen_at"],
    )
    op.create_index(
        "ix_event_artifacts_event_id",
        "event_artifacts",
        ["event_id"],
    )
    op.create_index(
        "ix_event_artifacts_kind",
        "event_artifacts",
        ["artifact_kind"],
    )
    op.create_index(
        "ix_event_artifacts_deleted_at",
        "event_artifacts",
        ["deleted_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_event_artifacts_deleted_at", table_name="event_artifacts")
    op.drop_index("ix_event_artifacts_kind", table_name="event_artifacts")
    op.drop_index("ix_event_artifacts_event_id", table_name="event_artifacts")
    op.drop_index("ix_detection_events_first_seen_at", table_name="detection_events")
    op.drop_index("ix_detection_events_event_state", table_name="detection_events")
    op.drop_index("ix_detection_events_capture_session", table_name="detection_events")
    op.drop_index("ix_detection_events_camera_first_seen_at", table_name="detection_events")
    op.drop_index("ix_capture_sessions_model_started_at", table_name="capture_sessions")
    op.drop_index("ix_capture_sessions_camera_started_at", table_name="capture_sessions")
    op.drop_table("event_artifacts")
    op.drop_table("detection_events")
    op.drop_table("capture_sessions")
