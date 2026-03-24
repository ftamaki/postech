from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.application.status_usecase import GetStatusUseCase, UpdateStatusUseCase
from app.domain.entities import Analysis, AnalysisStatus


def make_analysis(**kwargs) -> Analysis:
    defaults = dict(
        id=uuid4(),
        original_filename="diagram.png",
        file_path="/uploads/diagram.png",
        file_type="image/png",
        status=AnalysisStatus.RECEBIDO,
        error_message=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Analysis(**defaults)


@pytest.mark.asyncio
async def test_get_status_retorna_analise_existente():
    repo = AsyncMock()
    analysis_id = uuid4()
    expected = make_analysis(id=analysis_id, status=AnalysisStatus.EM_PROCESSAMENTO)
    repo.find_by_id.return_value = expected

    use_case = GetStatusUseCase(repo)
    result = await use_case.execute(analysis_id)

    assert result is not None
    assert result.status == AnalysisStatus.EM_PROCESSAMENTO
    repo.find_by_id.assert_called_once_with(analysis_id)


@pytest.mark.asyncio
async def test_get_status_retorna_none_para_id_inexistente():
    repo = AsyncMock()
    repo.find_by_id.return_value = None

    use_case = GetStatusUseCase(repo)
    result = await use_case.execute(uuid4())

    assert result is None


@pytest.mark.asyncio
async def test_update_status_atualiza_para_analisado():
    repo = AsyncMock()
    analysis_id = uuid4()
    updated = make_analysis(id=analysis_id, status=AnalysisStatus.ANALISADO)
    repo.update_status.return_value = updated

    use_case = UpdateStatusUseCase(repo)
    result = await use_case.execute(analysis_id, AnalysisStatus.ANALISADO)

    assert result.status == AnalysisStatus.ANALISADO
    repo.update_status.assert_called_once_with(analysis_id, AnalysisStatus.ANALISADO, None)


@pytest.mark.asyncio
async def test_update_status_salva_mensagem_de_erro():
    repo = AsyncMock()
    analysis_id = uuid4()
    updated = make_analysis(
        id=analysis_id,
        status=AnalysisStatus.ERRO,
        error_message="Timeout na API de IA",
    )
    repo.update_status.return_value = updated

    use_case = UpdateStatusUseCase(repo)
    result = await use_case.execute(
        analysis_id, AnalysisStatus.ERRO, "Timeout na API de IA"
    )

    assert result.status == AnalysisStatus.ERRO
    assert result.error_message == "Timeout na API de IA"
