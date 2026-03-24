import logging
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .application.status_usecase import GetStatusUseCase, UpdateStatusUseCase
from .application.upload_usecase import UploadUseCase
from .config import settings
from .domain.entities import AnalysisStatus
from .infrastructure.database import get_session, init_db
from .infrastructure.file_storage import LocalFileStorage
from .infrastructure.message_publisher import RabbitMQPublisher
from .infrastructure.repositories import SqlAnalysisRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

publisher: RabbitMQPublisher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global publisher
    await init_db(settings.DATABASE_URL)
    publisher = await RabbitMQPublisher.create(settings.RABBITMQ_URL, settings.QUEUE_NAME)
    logger.info("Upload Service iniciado")
    yield
    if publisher:
        await publisher.close()
    logger.info("Upload Service encerrado")


app = FastAPI(
    title="Architecture Analyzer - Upload Service",
    description="Recebe diagramas e gerencia o status do pipeline de análise",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf"}


# --- Schemas ---

class AnalysisResponse(BaseModel):
    id: str
    original_filename: str
    status: str
    file_type: str
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class UpdateStatusRequest(BaseModel):
    status: str
    error_message: Optional[str] = None


# --- Dependências ---

def get_file_storage() -> LocalFileStorage:
    return LocalFileStorage(settings.UPLOADS_DIR)


async def get_repo(session: AsyncSession = Depends(get_session)) -> SqlAnalysisRepository:
    return SqlAnalysisRepository(session)


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "service": "upload-service"}


@app.post("/analyses", response_model=AnalysisResponse, status_code=201)
async def upload_analysis(
    file: UploadFile = File(...),
    repo: SqlAnalysisRepository = Depends(get_repo),
    storage: LocalFileStorage = Depends(get_file_storage),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo não suportado: {file.content_type}",
        )

    file_data = await file.read()
    if len(file_data) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio")

    use_case = UploadUseCase(repo, storage, publisher)
    analysis = await use_case.execute(file_data, file.filename, file.content_type)

    return AnalysisResponse(
        id=str(analysis.id),
        original_filename=analysis.original_filename,
        status=analysis.status.value,
        file_type=analysis.file_type,
        error_message=analysis.error_message,
        created_at=analysis.created_at.isoformat(),
        updated_at=analysis.updated_at.isoformat(),
    )


@app.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    repo: SqlAnalysisRepository = Depends(get_repo),
):
    try:
        uid = UUID(analysis_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido")

    use_case = GetStatusUseCase(repo)
    analysis = await use_case.execute(uid)

    if not analysis:
        raise HTTPException(status_code=404, detail="Análise não encontrada")

    return AnalysisResponse(
        id=str(analysis.id),
        original_filename=analysis.original_filename,
        status=analysis.status.value,
        file_type=analysis.file_type,
        error_message=analysis.error_message,
        created_at=analysis.created_at.isoformat(),
        updated_at=analysis.updated_at.isoformat(),
    )


@app.put("/analyses/{analysis_id}/status", response_model=AnalysisResponse)
async def update_status(
    analysis_id: str,
    body: UpdateStatusRequest,
    repo: SqlAnalysisRepository = Depends(get_repo),
):
    """Endpoint interno — chamado pelo processing-service para atualizar o status."""
    try:
        uid = UUID(analysis_id)
        status = AnalysisStatus(body.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    use_case = UpdateStatusUseCase(repo)
    analysis = await use_case.execute(uid, status, body.error_message)

    if not analysis:
        raise HTTPException(status_code=404, detail="Análise não encontrada")

    return AnalysisResponse(
        id=str(analysis.id),
        original_filename=analysis.original_filename,
        status=analysis.status.value,
        file_type=analysis.file_type,
        error_message=analysis.error_message,
        created_at=analysis.created_at.isoformat(),
        updated_at=analysis.updated_at.isoformat(),
    )
