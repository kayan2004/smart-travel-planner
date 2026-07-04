"""add destinations.tags column for weighted cluster tags

Revision ID: a7c3e5f19d02
Revises: 9e4d1f6a8b02
Create Date: 2026-07-04 00:00:02.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = 'a7c3e5f19d02'
down_revision: Union[str, Sequence[str], None] = '9e4d1f6a8b02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "destinations",
        sa.Column("tags", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("destinations", "tags")
