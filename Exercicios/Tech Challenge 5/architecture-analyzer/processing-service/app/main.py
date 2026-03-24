import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from .application.process_usecase import ProcessDiagramUseCase
from .config import settings
from .infrastructure.ai_client import ClaudeAIAnalyzer
from .infrastructure.message_consumer import RabbitMQConsumer
from .infrastructure.reports_client import HttpReportsClient
from .infrastructure.upload_client import HttpUploadClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

consumer_task: asyncio.Task | None = None
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global consumer_task, http_client

    http_client = httpx.AsyncClient(timeout=120.0)

    ai_analyzer = ClaudeAIAnalyzer(settings.ANTHROPIC_API_KEY, settings.CLAUDE_MODEL)
    reports_client = HttpReportsClient(settings.REPORTS_SERVICE_URL, http_client)
    upload_client = HttpUploadClient(settings.UPLOAD_SERVICE_URL, http_client)

    use_case = ProcessDiagramUseCase(ai_analyzer, reports_client, upload_client)

    consumer = RabbitMQConsumer(settings.RABBITMQ_URL, settings.QUEUE_NAME, use_case)
    consumer_task = asyncio.create_task(consumer.start())

    logger.info("Processing Service iniciado — consumindo fila")
    yield

    if consumer_task:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    if http_client:
        await http_client.aclose()

    logger.info("Processing Service encerrado")


app = FastAPI(
    title="Architecture Analyzer - Processing Service",
    description="Consome diagramas da fila, analisa com IA e gera relatórios",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "processing-service"}
