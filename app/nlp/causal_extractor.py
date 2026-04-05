"""Causal relationship extraction using lexical cue phrases and dependency parsing.

Scans each sentence for causal cue phrases (e.g. "because", "due to",
"led to") and splits the sentence into cause/effect clauses.  Results are
matched back against the extracted event list to produce CausalRelation
objects linking events by their sentence text.
"""

import asyncio
import re
from dataclasses import dataclass

import structlog

from app.nlp.event_extractor import ExtractedEvent
from app.nlp.ner import _get_nlp

logger = structlog.get_logger(__name__)


# Causal cue phrases — ordered most-specific to least-specific to avoid
# partial-match shadowing (e.g. "because of" must come before "because").
_CAUSAL_CUES: list[tuple[str, float]] = [
    ("as a result of",       0.95),
    ("as a consequence of",  0.95),
    ("owing to",             0.90),
    ("because of",           0.90),
    ("due to",               0.90),
    ("caused by",            0.90),
    ("which caused",         0.88),
    ("which led to",         0.88),
    ("which resulted in",    0.88),
    ("resulting in",         0.88),
    ("resulted in",          0.88),
    ("led to",               0.85),
    ("leading to",           0.85),
    ("because",              0.85),
    ("consequently",         0.80),
    ("therefore",            0.80),
    ("hence",                0.75),
    ("thus",                 0.75),
    ("so that",              0.75),
]

# Pre-compiled patterns for speed: (regex, confidence)
_CUE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE), conf)
    for phrase, conf in _CAUSAL_CUES
]

# Cues where the cause appears *after* the cue word in the sentence
# (backward-causal direction): "X happened because of Y" → cause=Y, effect=X
_BACKWARD_CUES: frozenset[str] = frozenset(
    {
        "because",
        "because of",
        "due to",
        "owing to",
        "as a result of",
        "as a consequence of",
        "caused by",
    }
)


@dataclass
class CausalRelation:
    """A directional causal relationship extracted from a single sentence."""

    cause: str          # Text of the cause clause
    effect: str         # Text of the effect clause
    cue_phrase: str     # Trigger phrase that signalled the causal link
    confidence: float   # Extraction confidence [0.0 – 1.0]


def _find_cue(
    sentence: str,
) -> tuple[re.Match[str], float] | tuple[None, float]:
    """Return the first matching causal cue and its confidence, or (None, 0)."""
    for pattern, conf in _CUE_PATTERNS:
        m = pattern.search(sentence)
        if m:
            return m, conf
    return None, 0.0


def _split_on_cue(sentence: str, match: re.Match[str]) -> tuple[str, str]:
    """Split *sentence* into (cause, effect) using the matched cue position.

    Forward cues  ("led to", "resulted in", …): cause=before, effect=after.
    Backward cues ("because", "due to", …)    : cause=after,  effect=before.
    """
    cue_text = match.group(0).lower().strip()
    before = sentence[: match.start()].strip().rstrip(".,;:")
    after = sentence[match.end() :].strip().rstrip(".,;:")

    if cue_text in _BACKWARD_CUES:
        return after, before  # cause, effect
    return before, after      # cause, effect


def extract_causal_relations_sync(
    text: str,
    events: list[ExtractedEvent] | None = None,
) -> list[CausalRelation]:
    """Extract causal relations from *text* via lexical cue phrase detection.

    For each sentence that contains a recognised causal cue the sentence is
    split into cause and effect clauses.  When *events* is provided a relation
    is only emitted if at least one clause overlaps with a known event sentence,
    which tightens precision.

    Args:
        text: Coreference-resolved, normalised document text.
        events: Optional ExtractedEvent list from event_extractor.

    Returns:
        List of CausalRelation objects.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    # Build a lookup of known event sentences for precision filtering.
    event_sentences: set[str] = (
        {e.sentence.strip() for e in events} if events else set()
    )

    relations: list[CausalRelation] = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        match, confidence = _find_cue(sent_text)
        if match is None:
            continue

        cause, effect = _split_on_cue(sent_text, match)

        if not cause or not effect:
            continue  # degenerate split — nothing useful on one side

        # When we have an event list, require at least one side to overlap
        # with a known event sentence (reduces stray false positives).
        if event_sentences:
            cause_linked = any(
                cause in es or es in cause for es in event_sentences
            )
            effect_linked = any(
                effect in es or es in effect for es in event_sentences
            )
            if not cause_linked and not effect_linked:
                continue

        relations.append(
            CausalRelation(
                cause=cause,
                effect=effect,
                cue_phrase=match.group(0).strip(),
                confidence=round(confidence, 4),
            )
        )

    logger.debug("causal_relations_extracted", count=len(relations))
    return relations


async def extract_causal_relations(
    text: str,
    events: list[ExtractedEvent] | None = None,
) -> list[CausalRelation]:
    """Async wrapper for extract_causal_relations_sync (offloads to thread pool).

    Args:
        text: Coreference-resolved, normalised document text.
        events: Optional ExtractedEvent list.

    Returns:
        List of CausalRelation objects.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, extract_causal_relations_sync, text, events)
