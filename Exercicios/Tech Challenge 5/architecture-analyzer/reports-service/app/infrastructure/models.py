from datetime import datetime, timezone

from sqlalchemy import DateTime, Text
from sqlalchemy.dialects.postgresql import JSON, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class ReportModel(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(PGUUID(as_uuid=False), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(PGUUID(as_uuid=False), nullable=False, unique=True, index=True)
    components: Mapped[dict] = mapped_column(JSON, nullable=False, default=list)
    risks: Mapped[dict] = mapped_column(JSON, nullable=False, default=list)
    recommendations: Mapped[dict] = mapped_column(JSON, nullable=False, default=list)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    raw_ai_response: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
