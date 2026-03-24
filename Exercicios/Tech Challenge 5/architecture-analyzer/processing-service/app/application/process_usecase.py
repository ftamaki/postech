import logging

from ..domain.entities import ProcessingJob
from ..domain.ports import AIAnalyzer, ReportsClient, UploadClient

logger = logging.getLogger(__name__)

STATUS_EM_PROCESSAMENTO = "Em processamento"
STATUS_ANALISADO = "Analisado"
STATUS_ERRO = "Erro"


class ProcessDiagramUseCase:
    def __init__(
        self,
        ai_analyzer: AIAnalyzer,
        reports_client: ReportsClient,
        upload_client: UploadClient,
    ):
        self._ai = ai_analyzer
        self._reports = reports_client
        self._upload = upload_client

    async def execute(self, job: ProcessingJob) -> None:
        analysis_id = str(job.analysis_id)
        logger.info(f"Iniciando processamento: {analysis_id}")

        # 1. Atualiza status → Em processamento
        await self._upload.update_status(analysis_id, STATUS_EM_PROCESSAMENTO)

        try:
            # 2. Chama a IA
            result = await self._ai.analyze(job)

            # 3. Persiste o relatório
            await self._reports.create_report(result)

            # 4. Atualiza status → Analisado
            await self._upload.update_status(analysis_id, STATUS_ANALISADO)

            logger.info(f"Processamento concluído: {analysis_id}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Erro no processamento {analysis_id}: {error_msg}")

            # Garante que o status de erro seja registrado mesmo se a IA falhar
            try:
                await self._upload.update_status(analysis_id, STATUS_ERRO, error_msg)
            except Exception as update_err:
                logger.error(f"Falha ao registrar erro: {update_err}")

            raise
