"""NLP pipeline orchestrator.

Chains all NLP stages in order:
  1. Coreference resolution   (coref.py)
  2. Named entity recognition (ner.py)
  3. Temporal parsing         (temporal_parser.py)
  4. Event extraction         (event_extractor.py)
  5. Entity linking           (entity_linker.py)
  6. Causal extraction        (causal_extractor.py)

Returns a PipelineResult dataclass containing the outputs of every stage.
The ingestion normalizer (app/ingestion/normalizer.py) is expected to have
already cleaned the raw text before this pipeline is invoked.
"""

import asyncio
from dataclasses import dataclass, field

import structlog

from app.nlp.causal_extractor import CausalRelation, extract_causal_relations_sync
from app.nlp.coref import resolve_coref_sync
from app.nlp.entity_linker import LinkedEntity, link_entities_sync
from app.nlp.event_extractor import ExtractedEvent, extract_events_sync
from app.nlp.ner import NEREntity, extract_entities_sync
from app.nlp.temporal_parser import TemporalSpan, parse_temporal_entities_sync

logger = structlog.get_logger(__name__)


@dataclass
class PipelineResult:
    """Collected outputs from every NLP pipeline stage."""

    resolved_text: str                                             # After coref resolution
    entities: list[NEREntity] = field(default_factory=list)       # Stage 2
    temporal_spans: list[TemporalSpan] = field(default_factory=list)  # Stage 3
    events: list[ExtractedEvent] = field(default_factory=list)    # Stage 4
    linked_entities: list[LinkedEntity] = field(default_factory=list)  # Stage 5
    causal_relations: list[CausalRelation] = field(default_factory=list)  # Stage 6


def run_pipeline_sync(text: str) -> PipelineResult:
    """Run all NLP stages synchronously on *text*.

    Designed to be called inside a thread-pool executor (via run_pipeline)
    so that CPU-bound model calls do not block the async event loop.

    Args:
        text: Normalised text produced by app.ingestion.normalizer.

    Returns:
        PipelineResult with outputs from every stage.
    """
    log = logger.bind(text_length=len(text))

    # Stage 1: coreference resolution
    log.debug("pipeline_stage", stage="coref")
    resolved: str = resolve_coref_sync(text)

    # Stage 2: named entity recognition
    log.debug("pipeline_stage", stage="ner")
    entities: list[NEREntity] = extract_entities_sync(resolved)

    # Stage 3: temporal parsing — converts DATE/TIME entities to UTC spans
    log.debug("pipeline_stage", stage="temporal")
    temporal_spans: list[TemporalSpan] = parse_temporal_entities_sync(entities)

    # Stage 4: event extraction — SVO tuples with temporal attachment
    log.debug("pipeline_stage", stage="events")
    events: list[ExtractedEvent] = extract_events_sync(resolved, temporal_spans)

    # Stage 5: entity linking — intra-document clustering
    log.debug("pipeline_stage", stage="entity_linking")
    linked_entities: list[LinkedEntity] = link_entities_sync(entities)

    # Stage 6: causal relation extraction
    log.debug("pipeline_stage", stage="causal")
    causal_relations: list[CausalRelation] = extract_causal_relations_sync(
        resolved, events
    )

    log.info(
        "pipeline_complete",
        entities=len(entities),
        temporal_spans=len(temporal_spans),
        events=len(events),
        linked_entities=len(linked_entities),
        causal_relations=len(causal_relations),
    )

    return PipelineResult(
        resolved_text=resolved,
        entities=entities,
        temporal_spans=temporal_spans,
        events=events,
        linked_entities=linked_entities,
        causal_relations=causal_relations,
    )


async def run_pipeline(text: str) -> PipelineResult:
    """Async entry point for the NLP pipeline.

    Offloads the entire pipeline to a single thread-pool worker so that
    all six stages share the same OS thread and avoid GIL contention
    between concurrent executor submissions.

    Args:
        text: Normalised text produced by app.ingestion.normalizer.

    Returns:
        PipelineResult with outputs from every stage.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_pipeline_sync, text)
