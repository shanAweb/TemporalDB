"""
tests/unit/test_pipeline.py

Unit tests for app.nlp.pipeline.run_pipeline_sync().

All six NLP stage functions are mocked so no spaCy model or
sentence-transformer is loaded.  Tests verify stage ordering,
output propagation, and correct population of PipelineResult fields.

Coverage
--------
  - run_pipeline_sync: all six stages called in order
  - run_pipeline_sync: resolved text from coref passed to downstream stages
  - run_pipeline_sync: NER entities passed to temporal parser and linker
  - run_pipeline_sync: temporal spans passed to event extractor
  - run_pipeline_sync: PipelineResult fields populated from stage outputs
  - run_pipeline_sync: empty text produces empty PipelineResult fields
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from app.nlp.causal_extractor import CausalRelation
from app.nlp.entity_linker import LinkedEntity
from app.nlp.event_extractor import ExtractedEvent
from app.nlp.ner import NEREntity
from app.nlp.pipeline import PipelineResult, run_pipeline_sync
from app.nlp.temporal_parser import TemporalSpan


# ---------------------------------------------------------------------------
# Minimal stage-output stubs
# ---------------------------------------------------------------------------

def _ner_entity(text: str = "Acme Corp") -> NEREntity:
    return NEREntity(
        text=text, label="ORG", start_char=0, end_char=len(text), sentence=text
    )


def _temporal_span() -> MagicMock:
    span = MagicMock(spec=TemporalSpan)
    span.text = "Q3 2024"
    return span


def _extracted_event() -> MagicMock:
    ev = MagicMock(spec=ExtractedEvent)
    ev.sentence = "Revenue fell."
    return ev


def _linked_entity() -> MagicMock:
    ent = MagicMock(spec=LinkedEntity)
    ent.canonical_name = "Acme Corp"
    return ent


def _causal_relation() -> MagicMock:
    rel = MagicMock(spec=CausalRelation)
    rel.cause = "inflation"
    return rel


# ---------------------------------------------------------------------------
# Helper: run pipeline with every stage replaced by a controllable mock
# ---------------------------------------------------------------------------

def _run_with_mocks(
    *,
    resolved_text: str = "Revenue fell because costs rose.",
    entities=None,
    temporal_spans=None,
    events=None,
    linked_entities=None,
    causal_relations=None,
) -> tuple[PipelineResult, dict]:
    """
    Run run_pipeline_sync with all six stages mocked.

    Returns (PipelineResult, dict_of_mocks) so tests can assert call args.
    """
    entities        = entities        or [_ner_entity()]
    temporal_spans  = temporal_spans  or [_temporal_span()]
    events          = events          or [_extracted_event()]
    linked_entities = linked_entities or [_linked_entity()]
    causal_relations = causal_relations or [_causal_relation()]

    mocks: dict = {}

    with (
        patch("app.nlp.pipeline.resolve_coref_sync",
              return_value=resolved_text) as m_coref,
        patch("app.nlp.pipeline.extract_entities_sync",
              return_value=entities) as m_ner,
        patch("app.nlp.pipeline.parse_temporal_entities_sync",
              return_value=temporal_spans) as m_temporal,
        patch("app.nlp.pipeline.extract_events_sync",
              return_value=events) as m_events,
        patch("app.nlp.pipeline.link_entities_sync",
              return_value=linked_entities) as m_linking,
        patch("app.nlp.pipeline.extract_causal_relations_sync",
              return_value=causal_relations) as m_causal,
    ):
        result = run_pipeline_sync("Supply chain issues led to revenue decline.")
        mocks = {
            "coref": m_coref,
            "ner": m_ner,
            "temporal": m_temporal,
            "events": m_events,
            "linking": m_linking,
            "causal": m_causal,
        }

    return result, mocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunPipelineSyncStageOrder:
    def test_coref_called_with_original_text(self) -> None:
        _, mocks = _run_with_mocks()
        mocks["coref"].assert_called_once_with(
            "Supply chain issues led to revenue decline."
        )

    def test_ner_called_with_resolved_text(self) -> None:
        resolved = "Revenue fell because costs rose."
        _, mocks = _run_with_mocks(resolved_text=resolved)
        mocks["ner"].assert_called_once_with(resolved)

    def test_temporal_called_with_ner_entities(self) -> None:
        entities = [_ner_entity("Acme Corp"), _ner_entity("Q3 2024")]
        _, mocks = _run_with_mocks(entities=entities)
        mocks["temporal"].assert_called_once_with(entities)

    def test_events_called_with_resolved_text_and_spans(self) -> None:
        resolved = "Revenue fell because costs rose."
        spans = [_temporal_span()]
        _, mocks = _run_with_mocks(resolved_text=resolved, temporal_spans=spans)
        mocks["events"].assert_called_once_with(resolved, spans)

    def test_linking_called_with_ner_entities(self) -> None:
        entities = [_ner_entity()]
        _, mocks = _run_with_mocks(entities=entities)
        mocks["linking"].assert_called_once_with(entities)

    def test_causal_called_with_resolved_text_and_events(self) -> None:
        resolved = "Revenue fell because costs rose."
        events = [_extracted_event()]
        _, mocks = _run_with_mocks(resolved_text=resolved, events=events)
        mocks["causal"].assert_called_once_with(resolved, events)


class TestRunPipelineSyncResult:
    def test_resolved_text_in_result(self) -> None:
        resolved = "Costs rose because of inflation."
        result, _ = _run_with_mocks(resolved_text=resolved)
        assert result.resolved_text == resolved

    def test_entities_in_result(self) -> None:
        entities = [_ner_entity("Apple Inc.")]
        result, _ = _run_with_mocks(entities=entities)
        assert result.entities == entities

    def test_temporal_spans_in_result(self) -> None:
        spans = [_temporal_span()]
        result, _ = _run_with_mocks(temporal_spans=spans)
        assert result.temporal_spans == spans

    def test_events_in_result(self) -> None:
        events = [_extracted_event()]
        result, _ = _run_with_mocks(events=events)
        assert result.events == events

    def test_linked_entities_in_result(self) -> None:
        linked = [_linked_entity()]
        result, _ = _run_with_mocks(linked_entities=linked)
        assert result.linked_entities == linked

    def test_causal_relations_in_result(self) -> None:
        relations = [_causal_relation()]
        result, _ = _run_with_mocks(causal_relations=relations)
        assert result.causal_relations == relations

    def test_all_six_stages_called(self) -> None:
        _, mocks = _run_with_mocks()
        for name, mock in mocks.items():
            assert mock.called, f"Stage '{name}' was not called"


class TestRunPipelineSyncEmptyInputs:
    def test_empty_entities_propagates(self) -> None:
        result, _ = _run_with_mocks(entities=[])
        assert result.entities == []

    def test_empty_events_propagates(self) -> None:
        result, _ = _run_with_mocks(events=[])
        assert result.events == []

    def test_empty_causal_relations_propagates(self) -> None:
        result, _ = _run_with_mocks(causal_relations=[])
        assert result.causal_relations == []

    def test_returns_pipeline_result_type(self) -> None:
        result, _ = _run_with_mocks()
        assert isinstance(result, PipelineResult)
