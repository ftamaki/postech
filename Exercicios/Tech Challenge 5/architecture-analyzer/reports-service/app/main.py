import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .application.create_report_usecase import CreateReportUseCase
from .application.get_report_usecase import GetReportUseCase
from .config import settings
from .infrastructure.database import get_session, init_db
from .infrastructure.repositories import SqlReportRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(settings.DATABASE_URL)
    logger.info("Reports Service iniciado")
    yield
    logger.info("Reports Service encerrado")


app = FastAPI(
    title="Architecture Analyzer - Reports Service",
    description="Persiste e expõe relatórios técnicos de arquitetura",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Schemas ---

class ComponentSchema(BaseModel):
    name: str
    type: str
    description: str


class RiskSchema(BaseModel):
    severity: str
    title: str
    description: str


class RecommendationSchema(BaseModel):
    priority: str
    title: str
    description: str


class CreateReportRequest(BaseModel):
    analysis_id: str
    components: List[ComponentSchema]
    risks: List[RiskSchema]
    recommendations: List[RecommendationSchema]
    summary: str
    raw_ai_response: str


class ReportResponse(BaseModel):
    id: str
    analysis_id: str
    components: List[ComponentSchema]
    risks: List[RiskSchema]
    recommendations: List[RecommendationSchema]
    summary: str
    created_at: str


# --- Dependências ---

async def get_repo(session: AsyncSession = Depends(get_session)) -> SqlReportRepository:
    return SqlReportRepository(session)


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "service": "reports-service"}


@app.post("/reports", response_model=ReportResponse, status_code=201)
async def create_report(
    body: CreateReportRequest,
    repo: SqlReportRepository = Depends(get_repo),
):
    """Endpoint interno — chamado pelo processing-service após análise da IA."""
    try:
        analysis_id = UUID(body.analysis_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="analysis_id inválido")

    use_case = CreateReportUseCase(repo)
    report = await use_case.execute(
        analysis_id=analysis_id,
        components=[c.model_dump() for c in body.components],
        risks=[r.model_dump() for r in body.risks],
        recommendations=[rec.model_dump() for rec in body.recommendations],
        summary=body.summary,
        raw_ai_response=body.raw_ai_response,
    )

    return ReportResponse(
        id=str(report.id),
        analysis_id=str(report.analysis_id),
        components=[ComponentSchema(**{"name": c.name, "type": c.type, "description": c.description}) for c in report.components],
        risks=[RiskSchema(**{"severity": r.severity, "title": r.title, "description": r.description}) for r in report.risks],
        recommendations=[RecommendationSchema(**{"priority": rec.priority, "title": rec.title, "description": rec.description}) for rec in report.recommendations],
        summary=report.summary,
        created_at=report.created_at.isoformat(),
    )


@app.get("/reports/{analysis_id}", response_model=ReportResponse)
async def get_report(
    analysis_id: str,
    repo: SqlReportRepository = Depends(get_repo),
):
    try:
        uid = UUID(analysis_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido")

    use_case = GetReportUseCase(repo)
    report = await use_case.execute(uid)

    if not report:
        raise HTTPException(status_code=404, detail="Relatório não encontrado")

    return ReportResponse(
        id=str(report.id),
        analysis_id=str(report.analysis_id),
        components=[ComponentSchema(**{"name": c.name, "type": c.type, "description": c.description}) for c in report.components],
        risks=[RiskSchema(**{"severity": r.severity, "title": r.title, "description": r.description}) for r in report.risks],
        recommendations=[RecommendationSchema(**{"priority": rec.priority, "title": rec.title, "description": rec.description}) for rec in report.recommendations],
        summary=report.summary,
        created_at=report.created_at.isoformat(),
    )
