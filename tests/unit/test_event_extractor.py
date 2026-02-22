"""
tests/unit/test_event_extractor.py

Unit tests for app.nlp.event_extractor.

Tests the pure helper functions (_span_text, _collect_modifiers,
_attach_temporal) with minimal mock tokens, and tests
extract_events_sync() with a mocked spaCy model so no GPU or
transformer model is loaded.

Coverage
--------
  - _span_text: filters DET, PUNCT, and SPACE tokens from subtree
  - _collect_modifiers: collects prep/advmod/prt children
  - _attach_temporal: sentence-scoped match then document fallback
  - extract_events_sync: subject-verb extraction with temporal attach
  - extract_events_sync: sentences without a subject produce no events
  - extract_events_sync: no temporal spans leaves ts_start as None
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.nlp.event_extractor import (
    ExtractedEvent,
    _attach_temporal,
    _collect_modifiers,
    _span_text,
    extract_events_sync,
)
from app.nlp.temporal_parser import TemporalSpan


# ---------------------------------------------------------------------------
# Helpers for building minimal mock spaCy tokens
# ---------------------------------------------------------------------------

def _tok(text: str, pos: str = "NOUN", is_space: bool = False) -> MagicMock:
    """Return a minimal mock spaCy Token."""
    t = MagicMock()
    t.text = text
    t.pos_ = pos
    t.is_space = is_space
    t.subtree = [t]     # default: subtree is just itself
    t.children = []
    t.dep_ = "dep"
    return t


def _span(text: str, tokens: list[MagicMock]) -> MagicMock:
    """Return a mock spaCy Span (sentence)."""
    s = MagicMock()
    s.text = text
    s.__iter__ = lambda self: iter(tokens)
    return s


# ---------------------------------------------------------------------------
# _span_text
# ---------------------------------------------------------------------------

class TestSpanText:
    def test_single_content_token(self) -> None:
        tok = _tok("Revenue", "NOUN")
        tok.subtree = [tok]
        assert _span_text(tok) == "Revenue"

    def test_det_token_filtered(self) -> None:
        det = _tok("the", "DET")
        noun = _tok("Revenue", "NOUN")
        parent = MagicMock()
        parent.subtree = [det, noun]
        result = _span_text(parent)
        assert "the" not in result
        assert "Revenue" in result

    def test_punct_token_filtered(self) -> None:
        noun = _tok("costs", "NOUN")
        dot = _tok(".", "PUNCT")
        parent = MagicMock()
        parent.subtree = [noun, dot]
        result = _span_text(parent)
        assert "." not in result
        assert "costs" in result

    def test_space_token_filtered(self) -> None:
        sp = _tok(" ", "SPACE", is_space=True)
        noun = _tok("profits", "NOUN")
        parent = MagicMock()
        parent.subtree = [sp, noun]
        result = _span_text(parent)
        assert result == "profits"

    def test_empty_subtree_returns_empty(self) -> None:
        parent = MagicMock()
        parent.subtree = []
        assert _span_text(parent) == ""

    def test_multiple_content_tokens_joined(self) -> None:
        t1 = _tok("supply", "NOUN")
        t2 = _tok("chain", "NOUN")
        parent = MagicMock()
        parent.subtree = [t1, t2]
        assert _span_text(parent) == "supply chain"


# ---------------------------------------------------------------------------
# _collect_modifiers
# ---------------------------------------------------------------------------

class TestCollectModifiers:
    def _make_verb(self, child_deps: list[str]) -> MagicMock:
        """Verb token whose children have the given dep_ labels."""
        children = []
        for dep in child_deps:
            child = _tok(dep, "ADP")     # text == dep label for easy assertion
            child.dep_ = dep
            child.subtree = [child]
            children.append(child)
        verb = MagicMock()
        verb.children = children
        return verb

    def test_prep_child_collected(self) -> None:
        verb = self._make_verb(["prep"])
        mods = _collect_modifiers(verb)
        assert len(mods) == 1

    def test_advmod_child_collected(self) -> None:
        verb = self._make_verb(["advmod"])
        mods = _collect_modifiers(verb)
        assert len(mods) == 1

    def test_prt_child_collected(self) -> None:
        verb = self._make_verb(["prt"])
        mods = _collect_modifiers(verb)
        assert len(mods) == 1

    def test_non_modifier_dep_ignored(self) -> None:
        verb = self._make_verb(["nsubj", "dobj"])
        mods = _collect_modifiers(verb)
        assert mods == []

    def test_multiple_modifiers(self) -> None:
        verb = self._make_verb(["prep", "advmod"])
        mods = _collect_modifiers(verb)
        assert len(mods) == 2

    def test_no_children_returns_empty(self) -> None:
        verb = MagicMock()
        verb.children = []
        assert _collect_modifiers(verb) == []


# ---------------------------------------------------------------------------
# _attach_temporal
# ---------------------------------------------------------------------------

def _make_event(sentence: str) -> ExtractedEvent:
    return ExtractedEvent(subject="Revenue", verb="fall", obj="", sentence=sentence)


def _make_span(text: str, year: int = 2024) -> TemporalSpan:
    ts = datetime(year, 7, 1, tzinfo=timezone.utc)
    return TemporalSpan(text=text, ts_start=ts, ts_end=ts, is_range=False)


class TestAttachTemporal:
    def test_no_spans_leaves_ts_none(self) -> None:
        event = _make_event("Revenue fell.")
        _attach_temporal([event], [], MagicMock())
        assert event.ts_start is None

    def test_sentence_scoped_match(self) -> None:
        # The span text "Q3 2024" appears in the event sentence.
        event = _make_event("Revenue fell in Q3 2024.")
        span = _make_span("Q3 2024", year=2024)
        _attach_temporal([event], [span], MagicMock())
        assert event.ts_start == span.ts_start
        assert event.temporal_text == "Q3 2024"

    def test_document_fallback_when_no_sentence_match(self) -> None:
        # Span text "2023" does NOT appear in the sentence.
        event = _make_event("Revenue fell.")
        span = _make_span("2023", year=2023)
        _attach_temporal([event], [span], MagicMock())
        # Should still attach via document-level fallback (first span).
        assert event.ts_start is not None
        assert event.ts_start.year == 2023

    def test_sentence_match_wins_over_fallback(self) -> None:
        event = _make_event("Costs rose in Q3 2024.")
        q3_span   = _make_span("Q3 2024", year=2024)
        q1_span   = _make_span("Q1 2022", year=2022)
        _attach_temporal([event], [q1_span, q3_span], MagicMock())
        # q3_span text appears in sentence → must win over q1_span fallback.
        assert event.ts_start.year == 2024


# ---------------------------------------------------------------------------
# extract_events_sync (mocked spaCy)
# ---------------------------------------------------------------------------

def _make_mock_nlp_doc(sentences: list[MagicMock]) -> MagicMock:
    doc = MagicMock()
    doc.sents = iter(sentences)
    return doc


def _sentence_with_subject_verb(
    subj_text: str,
    verb_lemma: str,
    obj_text: str = "",
) -> MagicMock:
    """Build a mock sentence where one VERB ROOT has one nsubj child."""
    # Subject token
    subj = _tok(subj_text, "NOUN")
    subj.dep_ = "nsubj"
    subj.subtree = [subj]

    # Object token (optional)
    obj_tok = _tok(obj_text, "NOUN") if obj_text else None

    # Verb token
    verb = MagicMock()
    verb.pos_ = "VERB"
    verb.dep_ = "ROOT"
    verb.lemma_ = verb_lemma
    children = [subj]
    if obj_tok:
        obj_tok.dep_ = "dobj"
        obj_tok.subtree = [obj_tok]
        children.append(obj_tok)
    verb.children = children

    sent_text = f"{subj_text} {verb_lemma}" + (f" {obj_text}" if obj_text else "")
    return _span(sent_text, [subj, verb])


class TestExtractEventsSync:
    def _run(
        self,
        sentences: list[MagicMock],
        spans: list[TemporalSpan] | None = None,
    ) -> list[ExtractedEvent]:
        mock_nlp = MagicMock()
        mock_nlp.return_value = _make_mock_nlp_doc(sentences)
        with patch("app.nlp.event_extractor._get_nlp", return_value=mock_nlp):
            return extract_events_sync("any text", temporal_spans=spans)

    def test_subject_verb_extracted(self) -> None:
        sent = _sentence_with_subject_verb("Revenue", "fall")
        events = self._run([sent])
        assert len(events) == 1
        assert "Revenue" in events[0].subject
        assert events[0].verb == "fall"

    def test_subject_verb_object_extracted(self) -> None:
        sent = _sentence_with_subject_verb("Company", "report", "losses")
        events = self._run([sent])
        assert len(events) == 1
        assert "losses" in events[0].obj

    def test_no_subjects_produces_no_event(self) -> None:
        # A VERB ROOT with no children → no subjects → skipped.
        verb = MagicMock()
        verb.pos_ = "VERB"
        verb.dep_ = "ROOT"
        verb.lemma_ = "fall"
        verb.children = []
        sent = _span("fell.", [verb])
        events = self._run([sent])
        assert events == []

    def test_multiple_sentences(self) -> None:
        s1 = _sentence_with_subject_verb("Revenue", "fall")
        s2 = _sentence_with_subject_verb("Costs", "rise")
        events = self._run([s1, s2])
        assert len(events) == 2

    def test_no_temporal_spans_ts_start_none(self) -> None:
        sent = _sentence_with_subject_verb("Profits", "decline")
        events = self._run([sent], spans=None)
        assert events[0].ts_start is None

    def test_temporal_span_attached(self) -> None:
        # Span text matches sentence text → should attach.
        subj_text = "Revenue"
        verb_lemma = "fall"
        sent = _sentence_with_subject_verb(subj_text, verb_lemma)
        # Manually set sent.text so _attach_temporal can match.
        sent.text = f"{subj_text} {verb_lemma} in Q3 2024."
        span = _make_span("Q3 2024", year=2024)
        events = self._run([sent], spans=[span])
        assert events[0].ts_start is not None
