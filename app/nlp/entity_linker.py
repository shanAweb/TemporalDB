"""Entity linking: clusters NEREntity mentions within a document.

Resolves surface-form mentions to a canonical entity using a three-tier strategy:
  1. Exact string match (case-insensitive)          → confidence 1.0
  2. Fuzzy token overlap (difflib SequenceMatcher)  → confidence = ratio
  3. Embedding cosine similarity                    → confidence = similarity

Returns LinkedEntity objects carrying a cluster_id (UUID) that groups all
mentions of the same real-world entity.  Cross-document resolution against
the PostgreSQL/Neo4j store happens later in the storage layer (Phase 5).
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import structlog

from app.nlp.embedder import embed_batch_sync
from app.nlp.ner import NEREntity

logger = structlog.get_logger(__name__)

# Similarity thresholds
_FUZZY_THRESHOLD: float = 0.85      # SequenceMatcher ratio
_EMBEDDING_THRESHOLD: float = 0.92  # Cosine similarity (vectors are L2-normalised)


@dataclass
class LinkedEntity:
    """An NEREntity resolved to a canonical intra-document cluster."""

    text: str               # Original surface form as it appears in the text
    label: str              # spaCy entity type (ORG, PERSON, GPE, …)
    start_char: int         # Character offset start in the original text
    end_char: int           # Character offset end in the original text
    sentence: str           # Source sentence
    canonical_name: str     # Best/longest mention chosen for this cluster
    cluster_id: uuid.UUID   # Groups all mentions of the same real-world entity
    confidence: float       # Linking confidence [0.0 – 1.0]


@dataclass
class _Cluster:
    """Internal cluster built during the linking pass (not exported)."""

    cluster_id: uuid.UUID
    label: str
    canonical_name: str
    embedding: list[float]
    members: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    """Lowercase + strip for case-insensitive comparisons."""
    return text.lower().strip()


def _fuzzy(a: str, b: str) -> float:
    """SequenceMatcher similarity ratio between two strings."""
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _cosine(v1: list[float], v2: list[float]) -> float:
    """Dot product of two L2-normalised vectors (equals cosine similarity)."""
    return sum(a * b for a, b in zip(v1, v2))


def link_entities_sync(entities: list[NEREntity]) -> list[LinkedEntity]:
    """Cluster NEREntity mentions to canonical entities within a document.

    Two passes:
      Pass 1 — assign each mention to a cluster (or create a new one).
      Pass 2 — build LinkedEntity objects using the final canonical name
                (which may be updated mid-pass 1 to the longest mention seen).

    Entities of different spaCy label types are never merged.

    Args:
        entities: NEREntity list from extract_entities_sync().

    Returns:
        LinkedEntity list, one entry per input mention.
    """
    if not entities:
        return []

    # Embed all surface forms in a single batch to minimise model overhead.
    embeddings: list[list[float]] = embed_batch_sync([e.text for e in entities])

    clusters: list[_Cluster] = []
    # Parallel to `entities` — records which cluster each mention was assigned to
    # and the confidence of that assignment.
    assignments: list[tuple[_Cluster, float]] = []

    for entity, emb in zip(entities, embeddings):
        best_cluster: _Cluster | None = None
        best_score: float = 0.0

        for cluster in clusters:
            # Never merge mentions of different entity types.
            if cluster.label != entity.label:
                continue

            norm_text = _norm(entity.text)

            # ── Tier 1: exact match (case-insensitive) ──────────────────────
            if norm_text == _norm(cluster.canonical_name) or norm_text in (
                _norm(m) for m in cluster.members
            ):
                best_cluster = cluster
                best_score = 1.0
                break  # Can't do better than exact — short-circuit

            # ── Tier 2: fuzzy string similarity ─────────────────────────────
            ratio = _fuzzy(entity.text, cluster.canonical_name)
            if ratio >= _FUZZY_THRESHOLD and ratio > best_score:
                best_cluster = cluster
                best_score = ratio

            # ── Tier 3: embedding cosine similarity ─────────────────────────
            sim = _cosine(emb, cluster.embedding)
            if sim >= _EMBEDDING_THRESHOLD and sim > best_score:
                best_cluster = cluster
                best_score = sim

        if best_cluster is not None:
            best_cluster.members.append(entity.text)
            # Prefer the longest mention as the canonical name (more informative).
            if len(entity.text) > len(best_cluster.canonical_name):
                best_cluster.canonical_name = entity.text
            assignments.append((best_cluster, best_score))
        else:
            new_cluster = _Cluster(
                cluster_id=uuid.uuid4(),
                label=entity.label,
                canonical_name=entity.text,
                embedding=emb,
                members=[entity.text],
            )
            clusters.append(new_cluster)
            assignments.append((new_cluster, 1.0))

    # Pass 2: build output using final (possibly updated) canonical names.
    linked: list[LinkedEntity] = []
    for entity, (cluster, confidence) in zip(entities, assignments):
        linked.append(
            LinkedEntity(
                text=entity.text,
                label=entity.label,
                start_char=entity.start_char,
                end_char=entity.end_char,
                sentence=entity.sentence,
                canonical_name=cluster.canonical_name,
                cluster_id=cluster.cluster_id,
                confidence=round(confidence, 4),
            )
        )

    logger.debug(
        "entity_linking_complete",
        mention_count=len(entities),
        cluster_count=len(clusters),
    )
    return linked


async def link_entities(entities: list[NEREntity]) -> list[LinkedEntity]:
    """Async wrapper for link_entities_sync (offloads to thread pool).

    Args:
        entities: NEREntity list from extract_entities().

    Returns:
        LinkedEntity list, one entry per input mention.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, link_entities_sync, entities)
