"""Named Entity Recognition using spaCy.

Loads the configured spaCy model once and exposes extract_entities()
which returns structured NEREntity objects for downstream linking
and event extraction.
"""

import asyncio
from dataclasses import dataclass
from functools import lru_cache

import spacy
import structlog
from spacy.language import Language
from spacy.tokens import Doc

from app.config import settings

logger = structlog.get_logger(__name__)

# Entity types we care about — others are silently dropped
RELEVANT_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "GPE",       # Geopolitical entity (countries, cities)
    "LOC",       # Non-GPE locations
    "DATE",
    "TIME",
    "MONEY",
    "PERCENT",
    "PRODUCT",
    "EVENT",
    "LAW",
    "NORP",      # Nationalities, religious/political groups
}


@dataclass
class NEREntity:
    """A single named entity extracted from text."""

    text: str           # Surface form as it appears in the document
    label: str          # spaCy entity label (e.g. "ORG", "DATE")
    start_char: int     # Character offset start in the original text
    end_char: int       # Character offset end in the original text
    sentence: str       # The sentence containing this entity


@lru_cache(maxsize=1)
def _get_nlp() -> Language:
    """Load and cache the spaCy model (called once on first use)."""
    logger.info("spacy_model_loading", model=settings.spacy_model)
    nlp = spacy.load(settings.spacy_model)
    logger.info("spacy_model_loaded", model=settings.spacy_model)
    return nlp


def _process(text: str) -> Doc:
    """Run spaCy pipeline on text synchronously."""
    return _get_nlp()(text)


def extract_entities_sync(text: str) -> list[NEREntity]:
    """Extract named entities from text synchronously.

    Args:
        text: Normalized input text.

    Returns:
        List of NEREntity objects filtered to RELEVANT_ENTITY_TYPES.
    """
    doc = _process(text)

    # Build a char-offset → sentence mapping for quick lookup
    sent_map: dict[tuple[int, int], str] = {
        (sent.start_char, sent.end_char): sent.text for sent in doc.sents
    }

    def _find_sentence(start: int, end: int) -> str:
        for (s_start, s_end), s_text in sent_map.items():
            if s_start <= start and end <= s_end:
                return s_text
        return ""

    entities: list[NEREntity] = []
    for ent in doc.ents:
        if ent.label_ not in RELEVANT_ENTITY_TYPES:
            continue
        entities.append(
            NEREntity(
                text=ent.text.strip(),
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                sentence=_find_sentence(ent.start_char, ent.end_char),
            )
        )

    logger.debug("ner_extracted", entity_count=len(entities))
    return entities


async def extract_entities(text: str) -> list[NEREntity]:
    """Extract named entities asynchronously (offloads to thread pool).

    Args:
        text: Normalized input text.

    Returns:
        List of NEREntity objects.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, extract_entities_sync, text)
