from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.application.process_usecase import ProcessDiagramUseCase
from app.domain.entities import (
    AIAnalysisResult,
    Component,
    ProcessingJob,
    Recommendation,
    Risk,
)


def make_job(**kwargs) -> ProcessingJob:
    defaults = dict(
        analysis_id=uuid4(),
        file_path="/uploads/diagram.png",
        file_type="image/png",
        original_filename="diagram.png",
    )
    defaults.update(kwargs)
    return ProcessingJob(**defaults)


def make_result(analysis_id) -> AIAnalysisResult:
    return AIAnalysisResult(
        analysis_id=analysis_id,
        components=[Component(name="API Gateway", type="Gateway", description="Ponto de entrada")],
        risks=[Risk(severity="high", title="SPOF", description="Sem redundância")],
        recommendations=[Recommendation(priority="high", title="Adicionar réplicas", description="Scale horizontal")],
        summary="Arquitetura básica de microsserviços",
        raw_response='{"components": [], "risks": [], "recommendations": [], "summary": ""}',
    )


@pytest.mark.asyncio
async def test_execute_fluxo_feliz():
    ai = AsyncMock()
    reports = AsyncMock()
    upload = AsyncMock()

    job = make_job()
    ai.analyze.return_value = make_result(job.analysis_id)

    use_case = ProcessDiagramUseCase(ai, reports, upload)
    await use_case.execute(job)

    # Deve atualizar status para Em processamento e depois Analisado
    assert upload.update_status.call_count == 2
    calls = [c.args[1] for c in upload.update_status.call_args_list]
    assert calls[0] == "Em processamento"
    assert calls[1] == "Analisado"

    ai.analyze.assert_called_once_with(job)
    reports.create_report.assert_called_once()


@pytest.mark.asyncio
async def test_execute_falha_na_ia_atualiza_status_erro():
    ai = AsyncMock()
    reports = AsyncMock()
    upload = AsyncMock()

    ai.analyze.side_effect = ValueError("Resposta inválida da IA")

    use_case = ProcessDiagramUseCase(ai, reports, upload)

    with pytest.raises(ValueError):
        await use_case.execute(make_job())

    # Deve ter atualizado: Em processamento → Erro
    assert upload.update_status.call_count == 2
    last_status = upload.update_status.call_args_list[-1].args[1]
    assert last_status == "Erro"
    reports.create_report.assert_not_called()


@pytest.mark.asyncio
async def test_execute_falha_ao_salvar_relatorio_registra_erro():
    ai = AsyncMock()
    reports = AsyncMock()
    upload = AsyncMock()

    job = make_job()
    ai.analyze.return_value = make_result(job.analysis_id)
    reports.create_report.side_effect = Exception("reports-service indisponível")

    use_case = ProcessDiagramUseCase(ai, reports, upload)

    with pytest.raises(Exception, match="reports-service indisponível"):
        await use_case.execute(job)

    last_status = upload.update_status.call_args_list[-1].args[1]
    assert last_status == "Erro"
