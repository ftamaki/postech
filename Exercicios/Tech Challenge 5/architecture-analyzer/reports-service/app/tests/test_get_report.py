from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.application.get_report_usecase import GetReportUseCase
from app.domain.entities import Component, Recommendation, Report, Risk


def make_report(analysis_id=None) -> Report:
    return Report(
        id=uuid4(),
        analysis_id=analysis_id or uuid4(),
        components=[Component(name="API", type="Service", description="REST API")],
        risks=[Risk(severity="low", title="Sem TLS", description="Comunicação não criptografada")],
        recommendations=[Recommendation(priority="high", title="Adicionar TLS", description="Use HTTPS")],
        summary="Arquitetura simples",
        raw_ai_response="{}",
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_get_report_retorna_relatorio_existente():
    repo = AsyncMock()
    analysis_id = uuid4()
    expected = make_report(analysis_id)
    repo.find_by_analysis_id.return_value = expected

    use_case = GetReportUseCase(repo)
    result = await use_case.execute(analysis_id)

    assert result is not None
    assert result.analysis_id == analysis_id
    repo.find_by_analysis_id.assert_called_once_with(analysis_id)


@pytest.mark.asyncio
async def test_get_report_retorna_none_quando_nao_encontrado():
    repo = AsyncMock()
    repo.find_by_analysis_id.return_value = None

    use_case = GetReportUseCase(repo)
    result = await use_case.execute(uuid4())

    assert result is None
