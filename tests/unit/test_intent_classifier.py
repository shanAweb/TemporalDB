"""
tests/unit/test_intent_classifier.py

Unit tests for app.query.intent.

Tests the heuristic classifier (no LLM required) and the full
classify_intent() path with a mocked Ollama client.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.query.intent import Intent, IntentResult, _heuristic_classify, classify_intent


# ---------------------------------------------------------------------------
# _heuristic_classify — pure regex, no IO
# ---------------------------------------------------------------------------

class TestHeuristicClassify:
    # ── CAUSAL_WHY ────────────────────────────────────────────────────────
    @pytest.mark.parametrize("question", [
        "Why did revenue drop in Q3?",
        "why did the project fail?",
        "WHY did costs increase?",
        "What caused the supply chain disruption?",
        "What led to the revenue decline?",
        "Explain why profits fell last quarter.",
        "What was the reason for the delay?",
    ])
    def test_causal_why(self, question: str) -> None:
        result = _heuristic_classify(question)
        assert result is not None
        assert result.intent == Intent.CAUSAL_WHY
        assert result.method == "heuristic"
        assert result.confidence > 0.0

    # ── TEMPORAL_RANGE ────────────────────────────────────────────────────
    @pytest.mark.parametrize("question", [
        "What happened between July and September?",
        "Show events from January to March 2024.",
        "What occurred in Q3 2024?",
        "What happened in Q1?",
        "Events during last month.",
        "What happened in 2023?",
        "Show me events in July 2024.",
        "What happened last year?",
    ])
    def test_temporal_range(self, question: str) -> None:
        result = _heuristic_classify(question)
        assert result is not None
        assert result.intent == Intent.TEMPORAL_RANGE
        assert result.method == "heuristic"

    # ── SIMILARITY ────────────────────────────────────────────────────────
    @pytest.mark.parametrize("question", [
        "Find events similar to the supply chain disruption.",
        "Show me events like the Q3 revenue drop.",
        "Events related to the factory fire.",
        "Find events comparable to the 2023 shortage.",
        "Find events similar to last quarter's incident.",
    ])
    def test_similarity(self, question: str) -> None:
        result = _heuristic_classify(question)
        assert result is not None
        assert result.intent == Intent.SIMILARITY

    # ── ENTITY_TIMELINE ───────────────────────────────────────────────────
    @pytest.mark.parametrize("question", [
        "Show me the history of Acme Corp.",
        "Give me the timeline of Apple Inc.",
        "Show me everything about John Smith.",
        "What happened to Acme Corp?",
        "All events involving Microsoft.",
    ])
    def test_entity_timeline(self, question: str) -> None:
        result = _heuristic_classify(question)
        assert result is not None
        assert result.intent == Intent.ENTITY_TIMELINE

    # ── No match → None ───────────────────────────────────────────────────
    @pytest.mark.parametrize("question", [
        "List all documents.",
        "How many events are stored?",
        "Tell me about the database.",
    ])
    def test_no_heuristic_match(self, question: str) -> None:
        result = _heuristic_classify(question)
        assert result is None


# ---------------------------------------------------------------------------
# classify_intent — with mocked LLM
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    @pytest.mark.asyncio
    async def test_heuristic_short_circuits_llm(self) -> None:
        """Heuristic match should return without calling Ollama."""
        with patch("app.query.intent.ollama_client") as mock_client:
            result = await classify_intent("Why did revenue drop in Q3?")
        mock_client.generate.assert_not_called()
        assert result.intent == Intent.CAUSAL_WHY
        assert result.method == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_called_for_ambiguous_question(self) -> None:
        """Ambiguous questions should fall through to Ollama."""
        mock_generate = AsyncMock(return_value="TEMPORAL_RANGE")
        with patch("app.query.intent.ollama_client") as mock_client:
            mock_client.generate = mock_generate
            result = await classify_intent("Tell me about events.")
        mock_generate.assert_called_once()
        assert result.intent == Intent.TEMPORAL_RANGE
        assert result.method == "llm"
        assert result.confidence == pytest.approx(0.80)

    @pytest.mark.asyncio
    async def test_llm_unknown_label_falls_back_to_similarity(self) -> None:
        """Unrecognised LLM label should degrade gracefully to SIMILARITY."""
        mock_generate = AsyncMock(return_value="UNKNOWN_INTENT_LABEL")
        with patch("app.query.intent.ollama_client") as mock_client:
            mock_client.generate = mock_generate
            result = await classify_intent("Something completely ambiguous.")
        assert result.intent == Intent.SIMILARITY
        assert result.confidence == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_similarity(self) -> None:
        """LLM request failure should degrade gracefully to SIMILARITY."""
        mock_generate = AsyncMock(side_effect=RuntimeError("connection refused"))
        with patch("app.query.intent.ollama_client") as mock_client:
            mock_client.generate = mock_generate
            result = await classify_intent("Something completely ambiguous.")
        assert result.intent == Intent.SIMILARITY
        assert result.confidence == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_llm_label_case_insensitive(self) -> None:
        """LLM label matching should be case-insensitive."""
        mock_generate = AsyncMock(return_value="causal_why")
        with patch("app.query.intent.ollama_client") as mock_client:
            mock_client.generate = mock_generate
            result = await classify_intent("Something ambiguous.")
        assert result.intent == Intent.CAUSAL_WHY
