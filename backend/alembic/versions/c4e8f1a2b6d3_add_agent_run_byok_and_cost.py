"""add agent_runs.used_byok and estimated_cost_usd for the server-key free tier

Revision ID: c4e8f1a2b6d3
Revises: b1f4a9d3e7c2
Create Date: 2026-07-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c4e8f1a2b6d3'
down_revision: Union[str, Sequence[str], None] = 'b1f4a9d3e7c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "agent_runs",
        sa.Column(
            "used_byok",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    # Nullable: pre-existing rows have no captured cost. New rows always
    # write a float (0.0 for a free/unknown-priced model).
    op.add_column(
        "agent_runs",
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("agent_runs", "estimated_cost_usd")
    op.drop_column("agent_runs", "used_byok")
