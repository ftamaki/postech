import logging
from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4

from ..domain.entities import Component, Recommendation, Report, Risk
from ..domain.ports import ReportRepository

logger = logging.getLogger(__name__)


class CreateReportUseCase:
    def __init__(self, repo: ReportRepository):
        self._repo = repo

    async def execute(
        self,
        analysis_id: UUID,
        components: List[dict],
        risks: List[dict],
        recommendations: List[dict],
        summary: str,
        raw_ai_response: str,
    ) -> Report:
        report = Report(
            id=uuid4(),
            analysis_id=analysis_id,
            components=[Component(**c) for c in components],
            risks=[Risk(**r) for r in risks],
            recommendations=[Recommendation(**rec) for rec in recommendations],
            summary=summary,
            raw_ai_response=raw_ai_response,
            created_at=datetime.now(timezone.utc),
        )

        saved = await self._repo.save(report)
        logger.info(f"Relatório salvo: {saved.id} para análise {analysis_id}")
        return saved
