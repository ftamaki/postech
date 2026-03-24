from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.entities import Component, Recommendation, Report, Risk
from ..domain.ports import ReportRepository as ReportRepositoryPort
from .models import ReportModel


def _to_entity(model: ReportModel) -> Report:
    return Report(
        id=UUID(model.id),
        analysis_id=UUID(model.analysis_id),
        components=[Component(**c) for c in (model.components or [])],
        risks=[Risk(**r) for r in (model.risks or [])],
        recommendations=[Recommendation(**rec) for rec in (model.recommendations or [])],
        summary=model.summary,
        raw_ai_response=model.raw_ai_response,
        created_at=model.created_at,
    )


class SqlReportRepository(ReportRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, report: Report) -> Report:
        model = ReportModel(
            id=str(report.id),
            analysis_id=str(report.analysis_id),
            components=[{"name": c.name, "type": c.type, "description": c.description} for c in report.components],
            risks=[{"severity": r.severity, "title": r.title, "description": r.description} for r in report.risks],
            recommendations=[{"priority": rec.priority, "title": rec.title, "description": rec.description} for rec in report.recommendations],
            summary=report.summary,
            raw_ai_response=report.raw_ai_response,
            created_at=report.created_at,
        )
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return _to_entity(model)

    async def find_by_analysis_id(self, analysis_id: UUID) -> Optional[Report]:
        result = await self._session.execute(
            select(ReportModel).where(ReportModel.analysis_id == str(analysis_id))
        )
        model = result.scalar_one_or_none()
        return _to_entity(model) if model else None
