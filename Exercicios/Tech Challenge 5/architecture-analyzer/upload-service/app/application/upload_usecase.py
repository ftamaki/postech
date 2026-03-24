import logging
from datetime import datetime, timezone
from uuid import uuid4

from ..domain.entities import Analysis, AnalysisStatus
from ..domain.ports import AnalysisRepository, FileStorage, MessagePublisher

logger = logging.getLogger(__name__)


class UploadUseCase:
    def __init__(
        self,
        analysis_repo: AnalysisRepository,
        file_storage: FileStorage,
        message_publisher: MessagePublisher,
    ):
        self._repo = analysis_repo
        self._storage = file_storage
        self._publisher = message_publisher

    async def execute(
        self, file_data: bytes, filename: str, content_type: str
    ) -> Analysis:
        analysis_id = uuid4()

        file_path = await self._storage.save(file_data, str(analysis_id), content_type)

        now = datetime.now(timezone.utc)
        analysis = Analysis(
            id=analysis_id,
            original_filename=filename,
            file_path=file_path,
            file_type=content_type,
            status=AnalysisStatus.RECEBIDO,
            error_message=None,
            created_at=now,
            updated_at=now,
        )

        saved = await self._repo.save(analysis)

        await self._publisher.publish(
            {
                "analysis_id": str(analysis_id),
                "file_path": file_path,
                "file_type": content_type,
                "original_filename": filename,
            }
        )

        logger.info(f"Análise criada: {analysis_id} — arquivo: {filename}")
        return saved
