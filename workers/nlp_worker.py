"""NLP Worker — Kafka consumer entry point.

Consumes document.ingested events from Redpanda/Kafka and dispatches
each document through the NLP pipeline, then persists results to
PostgreSQL and syncs to Neo4j.

Run with:
    python -m workers.nlp_worker
"""

import asyncio
import json
import signal
import uuid

import structlog
from aiokafka import AIOKafkaConsumer

from app.config import settings
from app.database.neo4j import init_neo4j, close_neo4j, _get_driver
from app.database.postgres import async_session_factory, init_postgres, close_postgres, engine
from app.nlp.embedder import embed_sync
from app.nlp.pipeline import run_pipeline
from app.storage.entity_store import upsert_entity
from app.storage.event_store import insert_event, link_entities_to_event, insert_causal_relation
from app.storage.sync import sync_document

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

    text: str = payload.get("text", "")
    if not text:
        logger.warning("nlp_worker_empty_text", document_id=document_id)
        return

    # ── Run NLP pipeline ──────────────────────────────────────────────
    result = await run_pipeline(text)
    logger.info(
        "nlp_worker_pipeline_complete",
        document_id=document_id,
        entities=len(result.entities),
        events=len(result.events),
        linked_entities=len(result.linked_entities),
        causal_relations=len(result.causal_relations),
    )

    # ── Persist to PostgreSQL ─────────────────────────────────────────
    doc_uuid = uuid.UUID(document_id)

    async with async_session_factory() as pg_session:
        try:
            # 1. Upsert entities (deduplicated by canonical_name + type)
            #    Build a mapping from cluster_id -> Entity ORM object
            cluster_to_entity: dict[uuid.UUID, object] = {}
            for linked in result.linked_entities:
                entity = await upsert_entity(
                    pg_session,
                    name=linked.text,
                    canonical_name=linked.canonical_name,
                    entity_type=linked.label,
                )
                cluster_to_entity[linked.cluster_id] = entity

            # 2. Insert events and link them to entities
            #    Build a mapping from event sentence -> Event ORM for causal matching
            sentence_to_event_id: dict[str, uuid.UUID] = {}
            for ext_event in result.events:
                # Build description from SVO
                description = f"{ext_event.subject} {ext_event.verb}"
                if ext_event.obj:
                    description += f" {ext_event.obj}"

                # Generate embedding for the event description
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(None, embed_sync, description)

                event = await insert_event(
                    pg_session,
                    description=description,
                    event_type="action",  # default; could be refined
                    ts_start=ext_event.ts_start,
                    ts_end=ext_event.ts_end,
                    confidence=0.85,  # default pipeline confidence
                    source_sentence=ext_event.sentence,
                    embedding=embedding,
                    document_id=doc_uuid,
                )
                sentence_to_event_id[ext_event.sentence] = event.id

                # Link entities that appear in this event's sentence
                entity_ids_for_event: list[uuid.UUID] = []
                for linked in result.linked_entities:
                    if linked.sentence == ext_event.sentence:
                        entity_obj = cluster_to_entity.get(linked.cluster_id)
                        if entity_obj:
                            entity_ids_for_event.append(entity_obj.id)

                if entity_ids_for_event:
                    await link_entities_to_event(
                        pg_session,
                        event_id=event.id,
                        entity_ids=list(set(entity_ids_for_event)),
                    )

            # 3. Insert causal relations
            #    Match cause/effect clause text against event sentences
            for causal in result.causal_relations:
                cause_id = None
                effect_id = None
                for sent, eid in sentence_to_event_id.items():
                    if causal.cause in sent or sent in causal.cause:
                        cause_id = eid
                    if causal.effect in sent or sent in causal.effect:
                        effect_id = eid
                if cause_id and effect_id and cause_id != effect_id:
                    await insert_causal_relation(
                        pg_session,
                        cause_event_id=cause_id,
                        effect_event_id=effect_id,
                        confidence=causal.confidence,
                        evidence=f"{causal.cause} [{causal.cue_phrase}] {causal.effect}",
                    )

            await pg_session.commit()

            logger.info(
                "nlp_worker_persisted",
                document_id=document_id,
                events_stored=len(sentence_to_event_id),
                entities_stored=len(cluster_to_entity),
            )

            # ── Sync to Neo4j ─────────────────────────────────────────
            try:
                driver = _get_driver()
                async with driver.session() as neo4j_session:
                    sync_result = await sync_document(pg_session, neo4j_session, doc_uuid)
                    logger.info(
                        "nlp_worker_synced_neo4j",
                        document_id=document_id,
                        **sync_result.as_dict(),
                    )
            except Exception:
                logger.exception("nlp_worker_neo4j_sync_failed", document_id=document_id)

        except Exception:
            await pg_session.rollback()
            raise


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

    # Initialise database connections
    await init_postgres()
    await init_neo4j()

    consumer = await init_consumer()
    consume_task = asyncio.create_task(consume_loop(consumer))

    await shutdown_event.wait()
    consume_task.cancel()
    await consume_task

    # Clean up
    await close_neo4j()
    await close_postgres()
    logger.info("nlp_worker_stopped")


if __name__ == "__main__":
    asyncio.run(main())
