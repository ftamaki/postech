import logging

import httpx

from ..domain.entities import AIAnalysisResult
from ..domain.ports import ReportsClient

logger = logging.getLogger(__name__)


class HttpReportsClient(ReportsClient):
    def __init__(self, base_url: str, http_client: httpx.AsyncClient):
        self._base_url = base_url
        self._client = http_client

    async def create_report(self, result: AIAnalysisResult) -> None:
        payload = {
            "analysis_id": str(result.analysis_id),
            "components": [
                {"name": c.name, "type": c.type, "description": c.description}
                for c in result.components
            ],
            "risks": [
                {"severity": r.severity, "title": r.title, "description": r.description}
                for r in result.risks
            ],
            "recommendations": [
                {"priority": rec.priority, "title": rec.title, "description": rec.description}
                for rec in result.recommendations
            ],
            "summary": result.summary,
            "raw_ai_response": result.raw_response,
        }

        response = await self._client.post(f"{self._base_url}/reports", json=payload)
        response.raise_for_status()
        logger.info(f"Relatório criado para análise: {result.analysis_id}")
