from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.application.upload_usecase import UploadUseCase
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


@pytest.fixture
def mock_repo():
    return AsyncMock()


@pytest.fixture
def mock_storage():
    return AsyncMock()


@pytest.fixture
def mock_publisher():
    return AsyncMock()


@pytest.mark.asyncio
async def test_upload_cria_analise_com_status_recebido(mock_repo, mock_storage, mock_publisher):
    saved = make_analysis()
    mock_storage.save.return_value = "/uploads/diagram.png"
    mock_repo.save.return_value = saved

    use_case = UploadUseCase(mock_repo, mock_storage, mock_publisher)
    result = await use_case.execute(b"fake_data", "diagram.png", "image/png")

    assert result.status == AnalysisStatus.RECEBIDO
    mock_storage.save.assert_called_once()
    mock_repo.save.assert_called_once()
    mock_publisher.publish.assert_called_once()


@pytest.mark.asyncio
async def test_upload_publica_mensagem_com_campos_corretos(mock_repo, mock_storage, mock_publisher):
    analysis_id = uuid4()
    saved = make_analysis(id=analysis_id)
    mock_storage.save.return_value = "/uploads/test.png"
    mock_repo.save.return_value = saved

    use_case = UploadUseCase(mock_repo, mock_storage, mock_publisher)
    await use_case.execute(b"data", "test.png", "image/png")

    published_msg = mock_publisher.publish.call_args[0][0]
    assert "analysis_id" in published_msg
    assert "file_path" in published_msg
    assert "file_type" in published_msg
    assert "original_filename" in published_msg
    assert published_msg["file_type"] == "image/png"
    assert published_msg["original_filename"] == "test.png"


@pytest.mark.asyncio
async def test_upload_falha_ao_salvar_arquivo_nao_persiste(mock_repo, mock_storage, mock_publisher):
    mock_storage.save.side_effect = OSError("Disco cheio")

    use_case = UploadUseCase(mock_repo, mock_storage, mock_publisher)

    with pytest.raises(OSError):
        await use_case.execute(b"data", "diagram.png", "image/png")

    mock_repo.save.assert_not_called()
    mock_publisher.publish.assert_not_called()
