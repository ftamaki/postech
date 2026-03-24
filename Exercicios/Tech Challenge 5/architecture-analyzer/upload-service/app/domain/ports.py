from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from .entities import Analysis, AnalysisStatus


class AnalysisRepository(ABC):
    @abstractmethod
    async def save(self, analysis: Analysis) -> Analysis:
        pass

    @abstractmethod
    async def find_by_id(self, analysis_id: UUID) -> Optional[Analysis]:
        pass

    @abstractmethod
    async def update_status(
        self,
        analysis_id: UUID,
        status: AnalysisStatus,
        error_message: Optional[str] = None,
    ) -> Optional[Analysis]:
        pass


class FileStorage(ABC):
    @abstractmethod
    async def save(self, file_data: bytes, filename: str, content_type: str) -> str:
        """Salva o arquivo e retorna o caminho completo."""
        pass


class MessagePublisher(ABC):
    @abstractmethod
    async def publish(self, message: dict) -> None:
        pass
