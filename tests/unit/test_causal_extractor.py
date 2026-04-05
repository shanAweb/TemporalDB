"""
tests/unit/test_causal_extractor.py

Unit tests for app.nlp.causal_extractor.

Tests the pure helper functions (_find_cue, _split_on_cue) without
loading spaCy, and tests extract_causal_relations_sync with a mocked
spaCy model so no GPU/transformer is required.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from app.nlp.causal_extractor import (
    CausalRelation,
    _find_cue,
    _split_on_cue,
    extract_causal_relations_sync,
)


# ---------------------------------------------------------------------------
# _find_cue
# ---------------------------------------------------------------------------

class TestFindCue:
    def test_forward_cue_led_to(self) -> None:
        match, conf = _find_cue("Supply chain disruptions led to a revenue decline.")
        assert match is not None
        assert "led to" in match.group(0).lower()
        assert conf == pytest.approx(0.85)

    def test_backward_cue_due_to(self) -> None:
        match, conf = _find_cue("Revenue declined due to supply chain disruptions.")
        assert match is not None
        assert "due to" in match.group(0).lower()
        assert conf == pytest.approx(0.90)

    def test_high_confidence_cue(self) -> None:
        match, confidence = _find_cue("Profits fell as a result of rising costs.")
        assert match is not None
        assert confidence == pytest.approx(0.95)

    def test_no_cue(self) -> None:
        match, conf = _find_cue("The weather was pleasant on Tuesday.")
        assert match is None
        assert conf == 0.0

    def test_case_insensitive(self) -> None:
        match, conf = _find_cue("Costs rose BECAUSE OF inflation.")
        assert match is not None

    def test_most_specific_cue_wins(self) -> None:
        # "because of" must win over "because" (ordered most-specific first)
        match, conf = _find_cue("Sales fell because of weak demand.")
        assert match is not None
        assert match.group(0).lower().strip() == "because of"
        assert conf == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# _split_on_cue
# ---------------------------------------------------------------------------

class TestSplitOnCue:
    def _match(self, sentence: str, phrase: str) -> re.Match[str]:
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        m = pattern.search(sentence)
        assert m is not None, f"'{phrase}' not found in '{sentence}'"
        return m

    def test_forward_cue_led_to(self) -> None:
        sentence = "Supply disruptions led to a revenue decline."
        m = self._match(sentence, "led to")
        cause, effect = _split_on_cue(sentence, m)
        assert "disruptions" in cause
        assert "revenue" in effect

    def test_backward_cue_due_to(self) -> None:
        sentence = "Revenue declined due to supply disruptions."
        m = self._match(sentence, "due to")
        cause, effect = _split_on_cue(sentence, m)
        # "due to" is backward: cause = after cue, effect = before cue
        assert "disruptions" in cause
        assert "Revenue" in effect

    def test_backward_cue_because(self) -> None:
        sentence = "Sales dropped because demand was weak."
        m = self._match(sentence, "because")
        cause, effect = _split_on_cue(sentence, m)
        assert "demand was weak" in cause
        assert "Sales dropped" in effect

    def test_forward_cue_resulted_in(self) -> None:
        sentence = "Poor planning resulted in project failure."
        m = self._match(sentence, "resulted in")
        cause, effect = _split_on_cue(sentence, m)
        assert "Poor planning" in cause
        assert "failure" in effect


# ---------------------------------------------------------------------------
# extract_causal_relations_sync (with mocked spaCy)
# ---------------------------------------------------------------------------

def _make_mock_doc(sentences: list[str]) -> MagicMock:
    """Build a minimal mock spaCy Doc whose .sents yields mock Span objects."""
    doc = MagicMock()
    spans = []
    for sent_text in sentences:
        span = MagicMock()
        span.text = sent_text
        spans.append(span)
    doc.sents = iter(spans)
    return doc


class TestExtractCausalRelationsSync:
    def _run(self, text: str, sentences: list[str]) -> list[CausalRelation]:
        """Patch _get_nlp so the test never loads a real spaCy model."""
        mock_nlp = MagicMock()
        mock_nlp.return_value = _make_mock_doc(sentences)

        with patch("app.nlp.causal_extractor._get_nlp", return_value=mock_nlp):
            return extract_causal_relations_sync(text)

    def test_single_forward_relation(self) -> None:
        sentences = ["Supply chain disruptions led to a revenue decline."]
        relations = self._run(" ".join(sentences), sentences)
        assert len(relations) == 1
        rel = relations[0]
        assert "disruptions" in rel.cause
        assert "revenue" in rel.effect
        assert rel.confidence == pytest.approx(0.85)

    def test_single_backward_relation(self) -> None:
        sentences = ["Revenue fell because costs rose."]
        relations = self._run(" ".join(sentences), sentences)
        assert len(relations) == 1
        rel = relations[0]
        assert "costs rose" in rel.cause

    def test_no_cue_produces_no_relations(self) -> None:
        sentences = ["The sun rose at 6 a.m.", "Birds sang in the trees."]
        relations = self._run(" ".join(sentences), sentences)
        assert relations == []

    def test_multiple_sentences(self) -> None:
        sentences = [
            "Poor maintenance led to equipment failure.",
            "The weather was fine.",
            "Equipment failure resulted in production delays.",
        ]
        relations = self._run(" ".join(sentences), sentences)
        assert len(relations) == 2

    def test_degenerate_split_skipped(self) -> None:
        # A sentence where the cue is at the very start leaves an empty cause.
        sentences = ["Because demand fell."]
        relations = self._run(" ".join(sentences), sentences)
        # Should skip rather than emit an empty-cause relation.
        assert all(r.cause and r.effect for r in relations)
