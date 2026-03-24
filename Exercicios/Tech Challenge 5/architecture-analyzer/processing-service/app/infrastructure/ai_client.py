import base64
import json
import logging
from pathlib import Path

import anthropic

from ..domain.entities import (
    AIAnalysisResult,
    Component,
    ProcessingJob,
    Recommendation,
    Risk,
)
from ..domain.ports import AIAnalyzer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Você é um arquiteto de software especialista em análise de diagramas de arquitetura.
Sua tarefa é analisar o diagrama fornecido e gerar um relatório técnico estruturado.

REGRAS IMPORTANTES:
1. Analise SOMENTE o que está visível no diagrama — nunca invente componentes
2. Seja específico nos riscos — evite afirmações genéricas
3. Forneça recomendações acionáveis e práticas
4. Responda SEMPRE com JSON válido no formato especificado
5. Se não conseguir identificar o diagrama claramente, explique no campo summary
6. Mantenha as respostas em português brasileiro"""

ANALYSIS_PROMPT = """Analise este diagrama de arquitetura e gere um relatório estruturado.

Responda APENAS com JSON válido neste formato exato (sem markdown, sem texto extra):
{
  "components": [
    {"name": "string", "type": "string (ex: API, Database, Queue, Cache, Service, Load Balancer, Gateway)", "description": "string"}
  ],
  "risks": [
    {"severity": "high|medium|low", "title": "string", "description": "string"}
  ],
  "recommendations": [
    {"priority": "high|medium|low", "title": "string", "description": "string"}
  ],
  "summary": "Visão geral da arquitetura em 2-3 frases"
}"""


class ClaudeAIAnalyzer(AIAnalyzer):
    def __init__(self, api_key: str, model: str):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def analyze(self, job: ProcessingJob) -> AIAnalysisResult:
        logger.info(f"Iniciando análise de IA para: {job.analysis_id}")

        content = self._build_content(job)

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        raw_response = message.content[0].text
        logger.debug(f"Resposta bruta da IA: {raw_response[:200]}...")

        return self._parse_response(job.analysis_id, raw_response)

    def _build_content(self, job: ProcessingJob) -> list:
        file_data = Path(job.file_path).read_bytes()
        b64_data = base64.standard_b64encode(file_data).decode("utf-8")

        if job.file_type == "application/pdf":
            # Claude aceita PDF como document
            media_block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": b64_data,
                },
            }
        else:
            # Imagens: jpeg, png, gif, webp
            supported = {"image/jpeg", "image/png", "image/gif", "image/webp"}
            media_type = job.file_type if job.file_type in supported else "image/png"
            media_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            }

        return [media_block, {"type": "text", "text": ANALYSIS_PROMPT}]

    def _parse_response(self, analysis_id, raw: str) -> AIAnalysisResult:
        # Guardrail: extrai JSON mesmo se vier com texto extra
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"Resposta da IA não contém JSON válido: {raw[:200]}")

        data = json.loads(raw[start:end])

        components = [
            Component(**c)
            for c in data.get("components", [])
            if all(k in c for k in ("name", "type", "description"))
        ]
        risks = [
            Risk(**r)
            for r in data.get("risks", [])
            if all(k in r for k in ("severity", "title", "description"))
        ]
        recommendations = [
            Recommendation(**rec)
            for rec in data.get("recommendations", [])
            if all(k in rec for k in ("priority", "title", "description"))
        ]

        return AIAnalysisResult(
            analysis_id=analysis_id,
            components=components,
            risks=risks,
            recommendations=recommendations,
            summary=data.get("summary", ""),
            raw_response=raw,
        )
