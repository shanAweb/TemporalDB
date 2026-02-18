import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog
from aiokafka import AIOKafkaProducer

from app.config import settings

logger = structlog.get_logger(__name__)

_producer: AIOKafkaProducer | None = None


async def init_kafka_producer() -> None:
    """Start the Kafka producer (called on app startup)."""
    global _producer
    _producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )
    await _producer.start()
    logger.info("kafka_producer_started", servers=settings.kafka_bootstrap_servers)


async def close_kafka_producer() -> None:
    """Stop the Kafka producer (called on app shutdown)."""
    global _producer
    if _producer is not None:
        await _producer.stop()
        _producer = None
        logger.info("kafka_producer_stopped")


async def publish(
    topic: str,
    event_type: str,
    payload: dict[str, Any],
    key: str | None = None,
) -> None:
    """Publish a message to a Kafka topic.

    Args:
        topic: Target Kafka topic.
        event_type: Type of event (e.g. "document.uploaded").
        payload: Event data.
        key: Optional partition key (e.g. document_id).
    """
    if _producer is None:
        raise RuntimeError("Kafka producer not initialised. Call init_kafka_producer() first.")

    message = {
        "event_id": str(uuid4()),
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    await _producer.send_and_wait(topic, value=message, key=key)
    logger.debug("kafka_message_published", topic=topic, event_type=event_type)


async def publish_document_ingested(document_id: str, source: str, filename: str | None = None) -> None:
    """Publish a document.ingested event to trigger NLP processing."""
    await publish(
        topic=settings.kafka_ingestion_topic,
        event_type="document.ingested",
        payload={
            "document_id": document_id,
            "source": source,
            "filename": filename,
        },
        key=document_id,
    )


async def publish_document_processed(document_id: str, event_count: int, entity_count: int) -> None:
    """Publish a document.processed event after NLP pipeline completes."""
    await publish(
        topic=settings.kafka_ingestion_topic,
        event_type="document.processed",
        payload={
            "document_id": document_id,
            "event_count": event_count,
            "entity_count": entity_count,
        },
        key=document_id,
    )
