from abc import ABC, abstractmethod

from .entities import AIAnalysisResult, ProcessingJob


class AIAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, job: ProcessingJob) -> AIAnalysisResult:
        pass


class ReportsClient(ABC):
    @abstractmethod
    async def create_report(self, result: AIAnalysisResult) -> None:
        pass


class UploadClient(ABC):
    @abstractmethod
    async def update_status(
        self,
        analysis_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        pass
