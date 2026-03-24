from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.application.create_report_usecase import CreateReportUseCase
from app.domain.entities import Component, Recommendation, Report, Risk


def make_report(**kwargs) -> Report:
    analysis_id = kwargs.pop("analysis_id", uuid4())
    return Report(
        id=uuid4(),
        analysis_id=analysis_id,
        components=[Component(name="API", type="Service", description="REST API")],
        risks=[Risk(severity="high", title="SPOF", description="Sem redundância")],
        recommendations=[Recommendation(priority="high", title="Redundância", description="Adicionar réplicas")],
        summary="Arquitetura simples",
        raw_ai_response="{}",
        created_at=datetime.now(timezone.utc),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_create_report_salva_e_retorna():
    repo = AsyncMock()
    analysis_id = uuid4()
    expected = make_report(analysis_id=analysis_id)
    repo.save.return_value = expected

    use_case = CreateReportUseCase(repo)
    result = await use_case.execute(
        analysis_id=analysis_id,
        components=[{"name": "API", "type": "Service", "description": "REST API"}],
        risks=[{"severity": "high", "title": "SPOF", "description": "Sem redundância"}],
        recommendations=[{"priority": "high", "title": "Redundância", "description": "Adicionar réplicas"}],
        summary="Arquitetura simples",
        raw_ai_response="{}",
    )

    assert result.analysis_id == analysis_id
    repo.save.assert_called_once()


@pytest.mark.asyncio
async def test_create_report_mapeia_componentes_corretamente():
    repo = AsyncMock()
    analysis_id = uuid4()
    repo.save.return_value = make_report(analysis_id=analysis_id)

    use_case = CreateReportUseCase(repo)
    await use_case.execute(
        analysis_id=analysis_id,
        components=[
            {"name": "Gateway", "type": "API Gateway", "description": "Entrada"},
            {"name": "DB", "type": "Database", "description": "PostgreSQL"},
        ],
        risks=[],
        recommendations=[],
        summary="Test",
        raw_ai_response="",
    )

    saved_report = repo.save.call_args[0][0]
    assert len(saved_report.components) == 2
    assert saved_report.components[0].name == "Gateway"
    assert saved_report.components[1].type == "Database"
