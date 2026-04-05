"""
tests/unit/test_entity_linker.py

Unit tests for app.nlp.entity_linker.

Tests the pure helper functions (_fuzzy, _cosine, _norm) and the main
link_entities_sync() function with mocked embeddings so no sentence-
transformer model is loaded.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.nlp.entity_linker import (
    LinkedEntity,
    _cosine,
    _fuzzy,
    _norm,
    link_entities_sync,
)
from app.nlp.ner import NEREntity


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestNorm:
    def test_lowercases(self) -> None:
        assert _norm("Apple Inc.") == "apple inc."

    def test_strips_whitespace(self) -> None:
        assert _norm("  Apple  ") == "apple"


class TestFuzzy:
    def test_identical_strings(self) -> None:
        assert _fuzzy("Apple Inc.", "Apple Inc.") == pytest.approx(1.0)

    def test_completely_different(self) -> None:
        ratio = _fuzzy("Apple Inc.", "Microsoft Corp")
        assert ratio < 0.5

    def test_partial_overlap(self) -> None:
        ratio = _fuzzy("Apple Inc.", "Apple")
        assert 0.5 < ratio < 1.0

    def test_case_insensitive(self) -> None:
        assert _fuzzy("APPLE INC.", "apple inc.") == pytest.approx(1.0)


class TestCosine:
    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_partial_similarity(self) -> None:
        import math
        v1 = [1.0, 0.0]
        v2 = [math.sqrt(0.5), math.sqrt(0.5)]
        sim = _cosine(v1, v2)
        assert 0.5 < sim < 1.0


# ---------------------------------------------------------------------------
# link_entities_sync (mocked embeddings)
# ---------------------------------------------------------------------------

def _make_entity(text: str, label: str = "ORG") -> NEREntity:
    return NEREntity(
        text=text,
        label=label,
        start_char=0,
        end_char=len(text),
        sentence=f"Sentence containing {text}.",
    )


# Unit embedding: use identity-ish vectors so cosine similarity is predictable.
_MOCK_EMBEDDINGS: dict[str, list[float]] = {
    "Apple Inc.": [1.0, 0.0, 0.0],
    "Apple":      [1.0, 0.0, 0.0],   # identical → should cluster
    "Microsoft":  [0.0, 1.0, 0.0],   # orthogonal → separate cluster
    "John Smith": [0.0, 0.0, 1.0],
    "J. Smith":   [0.0, 0.0, 1.0],   # identical → should cluster
}


def _mock_embed_batch(texts: list[str]) -> list[list[float]]:
    return [_MOCK_EMBEDDINGS.get(t, [0.0, 0.0, 0.0]) for t in texts]


class TestLinkEntitiesSync:
    def _run(self, entities: list[NEREntity]) -> list[LinkedEntity]:
        with patch("app.nlp.entity_linker.embed_batch_sync", side_effect=_mock_embed_batch):
            return link_entities_sync(entities)

    def test_empty_input(self) -> None:
        assert self._run([]) == []

    def test_single_entity(self) -> None:
        entities = [_make_entity("Apple Inc.")]
        linked = self._run(entities)
        assert len(linked) == 1
        assert linked[0].canonical_name == "Apple Inc."
        assert linked[0].confidence == pytest.approx(1.0)

    def test_exact_match_clusters(self) -> None:
        # Same text appears twice → should be in the same cluster.
        entities = [_make_entity("Apple Inc."), _make_entity("Apple Inc.")]
        linked = self._run(entities)
        assert len(linked) == 2
        assert linked[0].cluster_id == linked[1].cluster_id

    def test_different_labels_not_merged(self) -> None:
        # ORG and PERSON with the same embedding should NOT merge.
        entities = [
            _make_entity("Apple Inc.", label="ORG"),
            _make_entity("Apple Inc.", label="PERSON"),
        ]
        linked = self._run(entities)
        assert linked[0].cluster_id != linked[1].cluster_id

    def test_high_embedding_similarity_clusters(self) -> None:
        # "Apple Inc." and "Apple" share the same mock embedding (cosine 1.0 ≥ 0.92).
        entities = [_make_entity("Apple Inc."), _make_entity("Apple")]
        linked = self._run(entities)
        assert linked[0].cluster_id == linked[1].cluster_id

    def test_dissimilar_entities_separate_clusters(self) -> None:
        # Orthogonal embeddings → separate clusters.
        entities = [_make_entity("Apple Inc."), _make_entity("Microsoft")]
        linked = self._run(entities)
        assert linked[0].cluster_id != linked[1].cluster_id

    def test_longest_mention_wins_canonical(self) -> None:
        # "Apple Inc." is longer than "Apple" — should become canonical.
        entities = [_make_entity("Apple"), _make_entity("Apple Inc.")]
        linked = self._run(entities)
        # Both should share a cluster whose canonical name is the longer mention.
        assert linked[0].cluster_id == linked[1].cluster_id
        canonical_names = {lk.canonical_name for lk in linked}
        assert "Apple Inc." in canonical_names

    def test_output_count_matches_input(self) -> None:
        entities = [
            _make_entity("Apple Inc."),
            _make_entity("Microsoft"),
            _make_entity("John Smith", label="PERSON"),
        ]
        linked = self._run(entities)
        assert len(linked) == len(entities)
