import logging
from typing import Optional

import httpx

from ..domain.ports import UploadClient

logger = logging.getLogger(__name__)


class HttpUploadClient(UploadClient):
    def __init__(self, base_url: str, http_client: httpx.AsyncClient):
        self._base_url = base_url
        self._client = http_client

    async def update_status(
        self,
        analysis_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        payload = {"status": status}
        if error_message:
            payload["error_message"] = error_message

        response = await self._client.put(
            f"{self._base_url}/analyses/{analysis_id}/status",
            json=payload,
        )
        response.raise_for_status()
        logger.info(f"Status atualizado: {analysis_id} → {status}")
