from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.entities import Analysis, AnalysisStatus
from ..domain.ports import AnalysisRepository as AnalysisRepositoryPort
from .models import AnalysisModel


def _to_entity(model: AnalysisModel) -> Analysis:
    return Analysis(
        id=UUID(model.id),
        original_filename=model.original_filename,
        file_path=model.file_path,
        file_type=model.file_type,
        status=AnalysisStatus(model.status),
        error_message=model.error_message,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


class SqlAnalysisRepository(AnalysisRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, analysis: Analysis) -> Analysis:
        model = AnalysisModel(
            id=str(analysis.id),
            original_filename=analysis.original_filename,
            file_path=analysis.file_path,
            file_type=analysis.file_type,
            status=analysis.status.value,
            error_message=analysis.error_message,
            created_at=analysis.created_at,
            updated_at=analysis.updated_at,
        )
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return _to_entity(model)

    async def find_by_id(self, analysis_id: UUID) -> Optional[Analysis]:
        result = await self._session.execute(
            select(AnalysisModel).where(AnalysisModel.id == str(analysis_id))
        )
        model = result.scalar_one_or_none()
        return _to_entity(model) if model else None

    async def update_status(
        self,
        analysis_id: UUID,
        status: AnalysisStatus,
        error_message: Optional[str] = None,
    ) -> Optional[Analysis]:
        now = datetime.now(timezone.utc)
        await self._session.execute(
            update(AnalysisModel)
            .where(AnalysisModel.id == str(analysis_id))
            .values(status=status.value, error_message=error_message, updated_at=now)
        )
        await self._session.commit()
        return await self.find_by_id(analysis_id)
