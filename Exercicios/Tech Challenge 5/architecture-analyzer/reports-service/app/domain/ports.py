from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from .entities import Report


class ReportRepository(ABC):
    @abstractmethod
    async def save(self, report: Report) -> Report:
        pass

    @abstractmethod
    async def find_by_analysis_id(self, analysis_id: UUID) -> Optional[Report]:
        pass
