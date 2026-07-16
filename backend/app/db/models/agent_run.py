from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="completed", nullable=False)
    # Whether this run used a caller-supplied (BYOK) key rather than the
    # server's. The free-tier gates (app/api/routes/agent_runs.py) only count
    # and budget-cap server-key runs - BYOK runs are the user's own spend.
    used_byok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Summed estimated dollar cost of this run's LLM calls (from the pricing
    # table in usage_logging.py). NULL only for legacy rows created before
    # this column existed; new rows always write a float (0.0 for a
    # free/unknown-priced model). Server-key rows' costs sum into the monthly
    # budget gate.
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="agent_runs")
    tool_logs = relationship("ToolLog", back_populates="agent_run", cascade="all, delete-orphan")
    recommendations = relationship(
        "Recommendation", back_populates="agent_run", cascade="all, delete-orphan"
    )
