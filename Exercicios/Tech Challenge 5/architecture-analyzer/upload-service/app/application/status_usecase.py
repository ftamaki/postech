import logging
from typing import Optional
from uuid import UUID

from ..domain.entities import Analysis, AnalysisStatus
from ..domain.ports import AnalysisRepository

logger = logging.getLogger(__name__)


class GetStatusUseCase:
    def __init__(self, analysis_repo: AnalysisRepository):
        self._repo = analysis_repo

    async def execute(self, analysis_id: UUID) -> Optional[Analysis]:
        return await self._repo.find_by_id(analysis_id)


class UpdateStatusUseCase:
    def __init__(self, analysis_repo: AnalysisRepository):
        self._repo = analysis_repo

    async def execute(
        self,
        analysis_id: UUID,
        status: AnalysisStatus,
        error_message: Optional[str] = None,
    ) -> Optional[Analysis]:
        updated = await self._repo.update_status(analysis_id, status, error_message)
        if updated:
            logger.info(f"Status atualizado: {analysis_id} → {status.value}")
        return updated
