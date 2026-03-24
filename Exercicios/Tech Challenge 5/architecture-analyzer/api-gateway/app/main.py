import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)
    logger.info("API Gateway iniciado")
    yield
    await http_client.aclose()
    logger.info("API Gateway encerrado")


app = FastAPI(
    title="Architecture Analyzer - API Gateway",
    description="Ponto de entrada unificado para o sistema de análise de arquiteturas",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "api-gateway"}


@app.post("/api/v1/analyses", summary="Enviar diagrama para análise")
async def upload_analysis(file: UploadFile = File(...)):
    """
    Recebe um diagrama (imagem ou PDF) e inicia o pipeline de análise.
    Retorna o ID da análise e o status inicial.
    """
    allowed_types = {"image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {file.content_type}. Use imagem (jpeg, png, gif, webp) ou PDF.",
        )

    file_data = await file.read()

    try:
        response = await http_client.post(
            f"{settings.UPLOAD_SERVICE_URL}/analyses",
            files={"file": (file.filename, file_data, file.content_type)},
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Erro no upload-service: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except httpx.RequestError as e:
        logger.error(f"Falha ao conectar com upload-service: {e}")
        raise HTTPException(status_code=503, detail="Serviço de upload indisponível")


@app.get("/api/v1/analyses/{analysis_id}", summary="Consultar status da análise")
async def get_analysis_status(analysis_id: str):
    """
    Retorna o status atual da análise: Recebido, Em processamento, Analisado ou Erro.
    """
    try:
        response = await http_client.get(
            f"{settings.UPLOAD_SERVICE_URL}/analyses/{analysis_id}"
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Análise não encontrada")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except httpx.RequestError as e:
        logger.error(f"Falha ao conectar com upload-service: {e}")
        raise HTTPException(status_code=503, detail="Serviço de upload indisponível")


@app.get("/api/v1/reports/{analysis_id}", summary="Obter relatório gerado")
async def get_report(analysis_id: str):
    """
    Retorna o relatório técnico com componentes identificados, riscos e recomendações.
    """
    try:
        response = await http_client.get(
            f"{settings.REPORTS_SERVICE_URL}/reports/{analysis_id}"
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Relatório ainda não disponível")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except httpx.RequestError as e:
        logger.error(f"Falha ao conectar com reports-service: {e}")
        raise HTTPException(status_code=503, detail="Serviço de relatórios indisponível")
