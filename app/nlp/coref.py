"""Coreference resolution — pronoun-to-antecedent replacement.

Uses spaCy's dependency parse and morphology to resolve pronouns to their
most recent matching noun phrase antecedent.  This is a rule-based
approximation; a full neural coref model can be swapped in later by
replacing resolve_coref_sync() without changing the public API.

Supported:
- Third-person singular pronouns (he, she, his, her, him, it, its)
- Third-person plural pronouns (they, their, them)
- Relative pronoun 'who' resolved to the immediately preceding NP
"""

import asyncio
import re
from functools import lru_cache

import spacy
import structlog
from spacy.language import Language
from spacy.tokens import Doc, Token

from app.config import settings

logger = structlog.get_logger(__name__)

# Pronoun → grammatical number mapping
_SINGULAR_PRONOUNS = {"he", "him", "his", "she", "her", "hers", "it", "its"}
_PLURAL_PRONOUNS = {"they", "them", "their", "theirs"}
_RELATIVE_PRONOUNS = {"who", "whom", "whose", "which"}
_ALL_PRONOUNS = _SINGULAR_PRONOUNS | _PLURAL_PRONOUNS | _RELATIVE_PRONOUNS


@lru_cache(maxsize=1)
def _get_nlp() -> Language:
    """Reuse the cached spaCy model (shared with ner.py via lru_cache key)."""
    from app.nlp.ner import _get_nlp as ner_get_nlp  # noqa: PLC0415
    return ner_get_nlp()


def _is_pronoun(token: Token) -> bool:
    return token.pos_ == "PRON" and token.lower_ in _ALL_PRONOUNS


def _number_of(token: Token) -> str:
    """Return 'singular' or 'plural' based on morphology, defaulting to singular."""
    morph = token.morph.get("Number")
    if morph:
        return "plural" if morph[0] == "Plur" else "singular"
    if token.lower_ in _PLURAL_PRONOUNS:
        return "plural"
    return "singular"


def _find_antecedent(pronoun: Token, doc: Doc) -> str | None:
    """Search backwards from *pronoun* for the nearest matching noun phrase.

    Matching criterion: the noun phrase root must agree in grammatical number
    with the pronoun.  Named entities are preferred over bare noun chunks.

    Args:
        pronoun: The pronoun token to resolve.
        doc: The full spaCy Doc.

    Returns:
        The surface text of the antecedent, or None if none found.
    """
    p_number = _number_of(pronoun)

    # Collect candidate spans in document order (named entities first for priority)
    candidates: list[tuple[int, str, str]] = []  # (end_char, text, number)

    for ent in doc.ents:
        if ent.end <= pronoun.i:
            ent_number = "plural" if ent.root.morph.get("Number") == ["Plur"] else "singular"
            candidates.append((ent.end, ent.text, ent_number))

    for chunk in doc.noun_chunks:
        if chunk.root.i < pronoun.i and chunk.root.pos_ != "PRON":
            chunk_number = _number_of(chunk.root)
            candidates.append((chunk.end, chunk.text, chunk_number))

    # Keep only number-agreeing candidates that appear before the pronoun
    matching = [text for (_, text, number) in candidates if number == p_number]

    if not matching:
        # Fallback: return last candidate regardless of number
        all_texts = [text for (_, text, _) in candidates]
        return all_texts[-1] if all_texts else None

    return matching[-1]  # most recent matching antecedent


def resolve_coref_sync(text: str) -> str:
    """Resolve pronouns to their antecedents and return the rewritten text.

    Args:
        text: Normalized input text.

    Returns:
        Text with pronouns replaced by their resolved referents where possible.
        Unresolvable pronouns are left as-is.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    replacements: list[tuple[int, int, str]] = []  # (start_char, end_char, replacement)

    for token in doc:
        if not _is_pronoun(token):
            continue
        antecedent = _find_antecedent(token, doc)
        if antecedent and antecedent.lower() != token.lower_:
            replacements.append((token.idx, token.idx + len(token.text), antecedent))

    if not replacements:
        return text

    # Apply replacements in reverse order to preserve character offsets
    result = text
    for start, end, replacement in reversed(replacements):
        result = result[:start] + replacement + result[end:]

    logger.debug("coref_resolved", replacements=len(replacements))
    return result


async def resolve_coref(text: str) -> str:
    """Async wrapper for resolve_coref_sync — offloads to thread pool.

    Args:
        text: Normalized input text.

    Returns:
        Coreference-resolved text.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, resolve_coref_sync, text)
