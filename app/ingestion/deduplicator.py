import hashlib
from dataclasses import dataclass

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)

_REDIS_KEY_PREFIX = "dedup:doc:"


@dataclass
class DeduplicationResult:
    """Result of a deduplication check."""

    is_duplicate: bool
    document_id: str  # existing ID if duplicate, new ID if fresh


def compute_fingerprint(text: str) -> str:
    """Return the SHA-256 hex digest of the normalized text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def check_and_register(
    text: str,
    document_id: str,
    redis: Redis,
) -> DeduplicationResult:
    """Check whether this text has been ingested before.

    If the fingerprint already exists in Redis, returns the original
    document_id as a duplicate.  Otherwise, registers the new fingerprint
    and returns the provided document_id as a fresh document.

    Args:
        text: Normalized document text to fingerprint.
        document_id: UUID string for the new document being ingested.
        redis: Async Redis client.

    Returns:
        DeduplicationResult indicating whether this is a duplicate
        and which document_id owns the content.
    """
    fingerprint = compute_fingerprint(text)
    key = f"{_REDIS_KEY_PREFIX}{fingerprint}"

    existing_id: str | None = await redis.get(key)

    if existing_id:
        logger.info(
            "duplicate_document_detected",
            fingerprint=fingerprint[:16],
            existing_document_id=existing_id,
        )
        return DeduplicationResult(is_duplicate=True, document_id=existing_id)

    await redis.set(key, document_id)
    logger.debug(
        "document_fingerprint_registered",
        fingerprint=fingerprint[:16],
        document_id=document_id,
    )
    return DeduplicationResult(is_duplicate=False, document_id=document_id)
