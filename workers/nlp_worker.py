"""NLP Worker — Kafka consumer entry point.

Consumes document.ingested events from Redpanda/Kafka and dispatches
each document through the NLP pipeline (implemented in Phase 4).

Run with:
    python -m workers.nlp_worker
"""

import asyncio
import json
import signal

import structlog
from aiokafka import AIOKafkaConsumer

from app.config import settings

logger = structlog.get_logger(__name__)

_consumer: AIOKafkaConsumer | None = None


async def init_consumer() -> AIOKafkaConsumer:
    """Create and start the Kafka consumer."""
    consumer = AIOKafkaConsumer(
        settings.kafka_ingestion_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    await consumer.start()
    logger.info(
        "kafka_consumer_started",
        topic=settings.kafka_ingestion_topic,
        group=settings.kafka_consumer_group,
        servers=settings.kafka_bootstrap_servers,
    )
    return consumer


async def process_message(message: dict) -> None:
    """Handle a single Kafka message envelope.

    Args:
        message: Deserialized message dict with keys:
                 event_id, event_type, timestamp, payload.
    """
    event_type: str = message.get("event_type", "")
    payload: dict = message.get("payload", {})
    document_id: str = payload.get("document_id", "")

    if event_type != "document.ingested":
        logger.debug("nlp_worker_skipping_event", event_type=event_type)
        return

    logger.info(
        "nlp_worker_received",
        document_id=document_id,
        source=payload.get("source"),
        filename=payload.get("filename"),
    )

    # ── Phase 4 hook ─────────────────────────────────────────────────────────
    # TODO: replace with actual NLP pipeline call, e.g.:
    #   await run_nlp_pipeline(document_id)
    logger.info("nlp_worker_pipeline_placeholder", document_id=document_id)
    # ─────────────────────────────────────────────────────────────────────────


async def consume_loop(consumer: AIOKafkaConsumer) -> None:
    """Main consumption loop — runs until cancelled."""
    logger.info("nlp_worker_consume_loop_started")
    try:
        async for msg in consumer:
            try:
                await process_message(msg.value)
            except Exception:
                logger.exception(
                    "nlp_worker_message_error",
                    topic=msg.topic,
                    partition=msg.partition,
                    offset=msg.offset,
                )
    except asyncio.CancelledError:
        logger.info("nlp_worker_consume_loop_cancelled")
    finally:
        await consumer.stop()
        logger.info("kafka_consumer_stopped")


async def main() -> None:
    """Entry point: start consumer and handle OS signals for clean shutdown."""
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _request_shutdown() -> None:
        logger.info("nlp_worker_shutdown_signal_received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)

    consumer = await init_consumer()
    consume_task = asyncio.create_task(consume_loop(consumer))

    await shutdown_event.wait()
    consume_task.cancel()
    await consume_task
    logger.info("nlp_worker_stopped")


if __name__ == "__main__":
    asyncio.run(main())
