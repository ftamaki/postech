import asyncio
import json
import logging
from uuid import UUID

import aio_pika

from ..application.process_usecase import ProcessDiagramUseCase
from ..domain.entities import ProcessingJob

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    def __init__(
        self,
        rabbitmq_url: str,
        queue_name: str,
        process_usecase: ProcessDiagramUseCase,
    ):
        self._rabbitmq_url = rabbitmq_url
        self._queue_name = queue_name
        self._use_case = process_usecase
        self._connection = None

    async def start(self) -> None:
        logger.info(f"Conectando ao RabbitMQ — fila: {self._queue_name}")
        self._connection = await aio_pika.connect_robust(self._rabbitmq_url)

        async with self._connection:
            channel = await self._connection.channel()
            await channel.set_qos(prefetch_count=1)  # Processa um por vez

            queue = await channel.declare_queue(self._queue_name, durable=True)
            logger.info("Aguardando mensagens...")

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process(requeue=False):
                        await self._handle_message(message.body)

    async def _handle_message(self, body: bytes) -> None:
        try:
            data = json.loads(body)
            job = ProcessingJob(
                analysis_id=UUID(data["analysis_id"]),
                file_path=data["file_path"],
                file_type=data["file_type"],
                original_filename=data["original_filename"],
            )
            await self._use_case.execute(job)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Mensagem inválida recebida: {e} — body: {body[:200]}")
        except Exception as e:
            # Erros da use case já foram tratados internamente (status atualizado)
            logger.error(f"Erro inesperado no handler: {e}")
