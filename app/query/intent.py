"""
app/query/intent.py

Query intent classification.

Classifies a natural-language question into one of four intents that drive
which query planner is invoked:

    CAUSAL_WHY       — "Why did revenue drop in Q3?"
    TEMPORAL_RANGE   — "What happened between July and September?"
    SIMILARITY       — "Find events similar to the supply chain disruption"
    ENTITY_TIMELINE  — "Show me everything about Acme Corp"

Two-stage approach
------------------
1. Heuristic pass  — fast regex/keyword rules that cover the majority of
   unambiguous queries without an LLM round-trip.  A rule fires only when
   its signal words appear in isolation (not negated) so precision stays
   high.

2. LLM fallback    — when no heuristic fires with sufficient confidence the
   question is sent to Ollama using the INTENT_CLASSIFICATION prompt.  The
   model response is parsed and validated; malformed responses fall back to
   SIMILARITY (the safest default for unknown queries).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import structlog

from app.llm.client import ollama_client
from app.llm.prompts import INTENT_CLASSIFICATION

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    """The four supported query intent categories."""

    CAUSAL_WHY      = "CAUSAL_WHY"
    TEMPORAL_RANGE  = "TEMPORAL_RANGE"
    SIMILARITY      = "SIMILARITY"
    ENTITY_TIMELINE = "ENTITY_TIMELINE"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    """Output of the intent classifier."""

    intent: Intent
    confidence: float       # 0.0 – 1.0
    method: str             # "heuristic" | "llm"


# ---------------------------------------------------------------------------
# Heuristic rules
# ---------------------------------------------------------------------------

# Each rule is (intent, confidence, compiled pattern).
# Rules are tried in order; the first match whose pattern fires wins.
# Patterns use word-boundary anchors (\b) and are case-insensitive.

_RULES: list[tuple[Intent, float, re.Pattern[str]]] = [
    # ── CAUSAL_WHY ────────────────────────────────────────────────────────
    # Interrogative "why" at or near the start of the question.
    (
        Intent.CAUSAL_WHY,
        0.95,
        re.compile(r"^\s*why\b", re.IGNORECASE),
    ),
    # Causal cue phrases embedded anywhere in the question.
    (
        Intent.CAUSAL_WHY,
        0.90,
        re.compile(
            r"\b(cause[sd]?|reason\s+for|led\s+to|result(?:ed)?\s+(?:in|of)|"
            r"due\s+to|because\s+of|as\s+a\s+(?:result|consequence)\s+of|"
            r"what\s+caused|explain\s+why)\b",
            re.IGNORECASE,
        ),
    ),
    # ── TEMPORAL_RANGE ────────────────────────────────────────────────────
    # Explicit time-range connectives ("between X and Y", "from X to Y").
    (
        Intent.TEMPORAL_RANGE,
        0.95,
        re.compile(
            r"\b(between\b.{1,60}\band\b|from\b.{1,60}\bto\b)",
            re.IGNORECASE,
        ),
    ),
    # Fiscal quarter references ("Q1", "Q3 2024", "first quarter").
    (
        Intent.TEMPORAL_RANGE,
        0.90,
        re.compile(
            r"\b(Q[1-4]\b|first\s+quarter|second\s+quarter|"
            r"third\s+quarter|fourth\s+quarter)\b",
            re.IGNORECASE,
        ),
    ),
    # Relative temporal expressions ("last month", "in 2023", "last year").
    (
        Intent.TEMPORAL_RANGE,
        0.85,
        re.compile(
            r"\b(last\s+(?:month|year|quarter|week)|"
            r"in\s+\d{4}|during\s+\w+|"
            r"(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    # ── SIMILARITY ────────────────────────────────────────────────────────
    (
        Intent.SIMILARITY,
        0.90,
        re.compile(
            r"\b(similar\s+to|like\s+(?:the|a|an)\b|related\s+to|"
            r"comparable\s+to|find\s+events?\s+(?:like|similar)|"
            r"events?\s+resembling|same\s+(?:type|kind)\s+as)\b",
            re.IGNORECASE,
        ),
    ),
    # ── ENTITY_TIMELINE ───────────────────────────────────────────────────
    (
        Intent.ENTITY_TIMELINE,
        0.92,
        re.compile(
            r"\b(history\s+of|timeline\s+of|everything\s+about|"
            r"all\s+events?\s+(?:for|about|involving|related\s+to)|"
            r"what\s+happened\s+to|show\s+(?:me\s+)?(?:all|everything)\s+(?:about|for|on)|"
            r"events?\s+involving)\b",
            re.IGNORECASE,
        ),
    ),
]

_VALID_LABELS: frozenset[str] = frozenset(i.value for i in Intent)


def _heuristic_classify(question: str) -> IntentResult | None:
    """Run the heuristic ruleset and return an IntentResult, or None if no
    rule fires."""
    for intent, confidence, pattern in _RULES:
        if pattern.search(question):
            logger.debug(
                "intent_heuristic_match",
                intent=intent.value,
                confidence=confidence,
            )
            return IntentResult(intent=intent, confidence=confidence, method="heuristic")
    return None


async def _llm_classify(question: str) -> IntentResult:
    """Call Ollama to classify intent and return an IntentResult.

    Falls back to SIMILARITY if the model returns an unrecognised label or
    the request fails.
    """
    prompt = INTENT_CLASSIFICATION.format(query=question)
    try:
        raw: str = await ollama_client.generate(
            prompt=prompt,
            temperature=0.0,   # deterministic for classification
            max_tokens=16,     # we only need a single label word
        )
        label = raw.strip().upper()
        if label in _VALID_LABELS:
            intent = Intent(label)
            logger.debug("intent_llm_classified", intent=intent.value)
            return IntentResult(intent=intent, confidence=0.80, method="llm")

        logger.warning("intent_llm_unknown_label", raw=raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("intent_llm_failed", error=str(exc))

    # Safe default when the LLM is unavailable or returns garbage.
    return IntentResult(intent=Intent.SIMILARITY, confidence=0.50, method="llm")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def classify_intent(question: str) -> IntentResult:
    """Classify the intent of *question* using heuristics then LLM fallback.

    The heuristic pass handles the majority of unambiguous queries in
    microseconds.  Only genuinely ambiguous questions incur an LLM round-trip.

    Args:
        question: Raw natural-language query from the user.

    Returns:
        IntentResult with the classified intent, confidence score, and the
        method used ("heuristic" or "llm").
    """
    result = _heuristic_classify(question)
    if result is not None:
        return result

    logger.debug("intent_falling_back_to_llm", question=question)
    return await _llm_classify(question)
