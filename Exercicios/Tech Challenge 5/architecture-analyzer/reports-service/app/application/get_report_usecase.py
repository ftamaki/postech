import logging
from typing import Optional
from uuid import UUID

from ..domain.entities import Report
from ..domain.ports import ReportRepository

logger = logging.getLogger(__name__)


class GetReportUseCase:
    def __init__(self, repo: ReportRepository):
        self._repo = repo

    async def execute(self, analysis_id: UUID) -> Optional[Report]:
        return await self._repo.find_by_analysis_id(analysis_id)
