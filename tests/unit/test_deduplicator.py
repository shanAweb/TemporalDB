"""
tests/unit/test_deduplicator.py

Unit tests for app.ingestion.deduplicator.

Tests the pure compute_fingerprint() helper and the async
check_and_register() function with a mocked Redis client so no
running Redis instance is required.

Coverage
--------
  - compute_fingerprint() returns consistent SHA-256 hex string
  - Same text always produces the same fingerprint
  - Different texts produce different fingerprints
  - check_and_register() returns fresh result when key absent
  - check_and_register() returns duplicate result when key present
  - Redis key follows the expected prefix + fingerprint format
  - New fingerprint is stored under the correct key
"""
from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock

import pytest

from app.ingestion.deduplicator import (
    DeduplicationResult,
    _REDIS_KEY_PREFIX,
    check_and_register,
    compute_fingerprint,
)


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------

class TestComputeFingerprint:
    def test_returns_hex_string(self) -> None:
        fp = compute_fingerprint("hello world")
        assert isinstance(fp, str)
        # SHA-256 hex digest is always 64 characters.
        assert len(fp) == 64

    def test_deterministic(self) -> None:
        text = "Supply chain disruptions led to a revenue decline."
        assert compute_fingerprint(text) == compute_fingerprint(text)

    def test_matches_manual_sha256(self) -> None:
        text = "The quarterly results exceeded expectations."
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert compute_fingerprint(text) == expected

    def test_different_texts_differ(self) -> None:
        fp1 = compute_fingerprint("text one")
        fp2 = compute_fingerprint("text two")
        assert fp1 != fp2

    def test_whitespace_sensitive(self) -> None:
        # Fingerprint is computed on the text as-is (no extra normalization here).
        assert compute_fingerprint("hello") != compute_fingerprint("hello ")

    def test_empty_string(self) -> None:
        fp = compute_fingerprint("")
        assert len(fp) == 64


# ---------------------------------------------------------------------------
# check_and_register
# ---------------------------------------------------------------------------

def _make_redis(existing_value: str | None = None) -> AsyncMock:
    """Return a Redis mock with get() returning *existing_value*."""
    redis = AsyncMock()
    redis.get.return_value = existing_value
    return redis


class TestCheckAndRegisterFresh:
    @pytest.mark.asyncio
    async def test_returns_not_duplicate(self) -> None:
        doc_id = str(uuid.uuid4())
        redis = _make_redis(existing_value=None)

        result = await check_and_register("unique text", doc_id, redis)

        assert result.is_duplicate is False
        assert result.document_id == doc_id

    @pytest.mark.asyncio
    async def test_fingerprint_stored_in_redis(self) -> None:
        doc_id = str(uuid.uuid4())
        redis = _make_redis(existing_value=None)
        text = "Unique document content for storage test."

        await check_and_register(text, doc_id, redis)

        expected_key = f"{_REDIS_KEY_PREFIX}{compute_fingerprint(text)}"
        redis.set.assert_awaited_once_with(expected_key, doc_id)

    @pytest.mark.asyncio
    async def test_redis_get_called_with_correct_key(self) -> None:
        doc_id = str(uuid.uuid4())
        redis = _make_redis(existing_value=None)
        text = "Another unique piece of text."

        await check_and_register(text, doc_id, redis)

        expected_key = f"{_REDIS_KEY_PREFIX}{compute_fingerprint(text)}"
        redis.get.assert_awaited_once_with(expected_key)


class TestCheckAndRegisterDuplicate:
    @pytest.mark.asyncio
    async def test_returns_is_duplicate_true(self) -> None:
        existing_id = str(uuid.uuid4())
        new_id = str(uuid.uuid4())
        redis = _make_redis(existing_value=existing_id)

        result = await check_and_register("repeated text", new_id, redis)

        assert result.is_duplicate is True

    @pytest.mark.asyncio
    async def test_returns_existing_document_id(self) -> None:
        existing_id = str(uuid.uuid4())
        new_id = str(uuid.uuid4())
        redis = _make_redis(existing_value=existing_id)

        result = await check_and_register("repeated text", new_id, redis)

        # Must return the *existing* id, not the new one.
        assert result.document_id == existing_id
        assert result.document_id != new_id

    @pytest.mark.asyncio
    async def test_no_set_called_for_duplicate(self) -> None:
        """Redis.set must NOT be called when a duplicate is detected."""
        existing_id = str(uuid.uuid4())
        redis = _make_redis(existing_value=existing_id)

        await check_and_register("repeated text", str(uuid.uuid4()), redis)

        redis.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deduplication_result_type(self) -> None:
        existing_id = str(uuid.uuid4())
        redis = _make_redis(existing_value=existing_id)

        result = await check_and_register("any text", str(uuid.uuid4()), redis)

        assert isinstance(result, DeduplicationResult)
