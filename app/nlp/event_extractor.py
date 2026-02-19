"""Event extraction using spaCy dependency parsing.

Walks the dependency tree of each sentence to extract Subject-Verb-Object
(SVO) tuples.  Each tuple is enriched with the nearest temporal span from
the document and the full source sentence for traceability.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from spacy.tokens import Doc, Span, Token

from app.nlp.ner import _get_nlp
from app.nlp.temporal_parser import TemporalSpan

logger = structlog.get_logger(__name__)

# Dependency labels that mark a subject
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}

# Dependency labels that mark an object
_OBJECT_DEPS = {"dobj", "pobj", "iobj", "attr", "oprd", "dative"}

# POS tags to skip when collecting noun phrase text
_SKIP_POS = {"DET", "PUNCT", "SPACE"}


@dataclass
class ExtractedEvent:
    """A single event extracted from a sentence."""

    subject: str                        # Who/what performs the action
    verb: str                           # The action (lemmatized root verb)
    obj: str                            # Who/what is acted upon (may be empty)
    sentence: str                       # Full source sentence
    ts_start: datetime | None = None    # Attached temporal start (if found)
    ts_end: datetime | None = None      # Attached temporal end (if found)
    temporal_text: str | None = None    # Original temporal expression text
    modifiers: list[str] = field(default_factory=list)  # Prepositional phrases


def _span_text(token: Token) -> str:
    """Collect the subtree of *token* as a readable string, skipping punct/det."""
    parts = [
        t.text for t in token.subtree
        if t.pos_ not in _SKIP_POS and not t.is_space
    ]
    return " ".join(parts).strip()


def _collect_modifiers(verb_token: Token) -> list[str]:
    """Gather prepositional and adverbial modifiers attached to the verb."""
    modifiers: list[str] = []
    for child in verb_token.children:
        if child.dep_ in ("prep", "advmod", "npadvmod", "prt"):
            modifiers.append(_span_text(child))
    return modifiers


def _extract_from_sentence(sent: Span) -> list[ExtractedEvent]:
    """Extract SVO tuples from a single sentence span."""
    events: list[ExtractedEvent] = []

    for token in sent:
        # Root verbs and auxiliary-less verbs in the sentence
        if token.pos_ != "VERB" or token.dep_ not in ("ROOT", "relcl", "advcl", "xcomp"):
            continue

        # Find subjects
        subjects = [
            child for child in token.children if child.dep_ in _SUBJECT_DEPS
        ]
        # Find objects
        objects = [
            child for child in token.children if child.dep_ in _OBJECT_DEPS
        ]

        if not subjects:
            continue  # Skip verbless or subjectless constructions

        subject_text = _span_text(subjects[0])
        verb_text = token.lemma_.lower()
        obj_text = _span_text(objects[0]) if objects else ""
        modifiers = _collect_modifiers(token)

        events.append(
            ExtractedEvent(
                subject=subject_text,
                verb=verb_text,
                obj=obj_text,
                sentence=sent.text.strip(),
                modifiers=modifiers,
            )
        )

    return events


def _attach_temporal(
    events: list[ExtractedEvent],
    spans: list[TemporalSpan],
    doc: Doc,
) -> None:
    """Attach the nearest TemporalSpan to each event in-place.

    Strategy: for each event, find the TemporalSpan whose source text
    appears in the same sentence.  Fall back to the document's first span.
    """
    if not spans:
        return

    for event in events:
        matched: TemporalSpan | None = None
        for span in spans:
            if span.text in event.sentence:
                matched = span
                break
        if matched is None:
            matched = spans[0]  # document-level fallback

        event.ts_start = matched.ts_start
        event.ts_end = matched.ts_end
        event.temporal_text = matched.text


def extract_events_sync(
    text: str,
    temporal_spans: list[TemporalSpan] | None = None,
) -> list[ExtractedEvent]:
    """Extract SVO events from *text* and attach temporal information.

    Args:
        text: Coreference-resolved, normalized document text.
        temporal_spans: Pre-parsed temporal spans from temporal_parser.

    Returns:
        List of ExtractedEvent objects.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    events: list[ExtractedEvent] = []
    for sent in doc.sents:
        events.extend(_extract_from_sentence(sent))

    if temporal_spans:
        _attach_temporal(events, temporal_spans, doc)

    logger.debug("events_extracted", count=len(events))
    return events


async def extract_events(
    text: str,
    temporal_spans: list[TemporalSpan] | None = None,
) -> list[ExtractedEvent]:
    """Async wrapper for extract_events_sync.

    Args:
        text: Coreference-resolved, normalized document text.
        temporal_spans: Pre-parsed temporal spans from temporal_parser.

    Returns:
        List of ExtractedEvent objects.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, extract_events_sync, text, temporal_spans
    )
