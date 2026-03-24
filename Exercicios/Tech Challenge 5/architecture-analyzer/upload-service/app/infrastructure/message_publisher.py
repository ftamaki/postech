import json
import logging

import aio_pika

from ..domain.ports import MessagePublisher as MessagePublisherPort

logger = logging.getLogger(__name__)


class RabbitMQPublisher(MessagePublisherPort):
    def __init__(self, connection: aio_pika.RobustConnection, queue_name: str):
        self._connection = connection
        self._queue_name = queue_name

    @classmethod
    async def create(cls, rabbitmq_url: str, queue_name: str) -> "RabbitMQPublisher":
        connection = await aio_pika.connect_robust(rabbitmq_url)
        publisher = cls(connection, queue_name)
        # Garante que a fila existe
        async with connection.channel() as channel:
            await channel.declare_queue(queue_name, durable=True)
        logger.info(f"Publicador RabbitMQ conectado — fila: {queue_name}")
        return publisher

    async def publish(self, message: dict) -> None:
        async with self._connection.channel() as channel:
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=self._queue_name,
            )
        logger.info(f"Mensagem publicada na fila {self._queue_name}: {message['analysis_id']}")

    async def close(self) -> None:
        await self._connection.close()
