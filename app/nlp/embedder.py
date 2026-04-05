"""Embedding generation using sentence-transformers.

Wraps the all-MiniLM-L6-v2 model (or whatever is configured in settings)
and exposes async-friendly helpers for single and batch embedding.
The model is loaded once at startup and reused across requests.
"""

import asyncio
from functools import lru_cache

import structlog
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load and cache the embedding model (called once on first use)."""
    logger.info("embedding_model_loading", model=settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info("embedding_model_loaded", model=settings.embedding_model)
    return model


def embed_sync(text: str) -> list[float]:
    """Embed a single text string synchronously.

    Args:
        text: Input text to embed.

    Returns:
        A list of floats of length settings.embedding_dimension (384).
    """
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch_sync(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts synchronously.

    Args:
        texts: List of input strings.

    Returns:
        List of embedding vectors, one per input text.
    """
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]


async def embed(text: str) -> list[float]:
    """Embed a single text string in a thread pool to avoid blocking the event loop.

    Args:
        text: Input text to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_sync, text)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in a thread pool to avoid blocking the event loop.

    Args:
        texts: List of input strings.

    Returns:
        List of embedding vectors.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_batch_sync, texts)
