"""AES-256-GCM credential encryption/decryption.

Credentials are stored as an opaque versioned string:
    v1:{base64_nonce}:{base64_ciphertext_with_tag}

The 16-byte GCM authentication tag is appended to the ciphertext by the
``cryptography`` library automatically, so the stored blob is tamper-evident.

The key is read from the ``CREDENTIALS_ENCRYPTION_KEY`` environment variable
(64 hex chars = 32 bytes). Generate one with:
    python -c "import secrets; print(secrets.token_hex(32))"
"""

import base64
import json
import os
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_VERSION = "v1"
_NONCE_SIZE = 12  # bytes — standard for AES-GCM


def _get_key() -> bytes:
    """Load the 32-byte AES key from the environment."""
    hex_key = os.environ.get("CREDENTIALS_ENCRYPTION_KEY", "")
    if not hex_key:
        raise RuntimeError(
            "CREDENTIALS_ENCRYPTION_KEY is not set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    try:
        key = bytes.fromhex(hex_key)
    except ValueError as exc:
        raise RuntimeError(
            "CREDENTIALS_ENCRYPTION_KEY must be a 64-character hex string."
        ) from exc
    if len(key) != 32:
        raise RuntimeError(
            f"CREDENTIALS_ENCRYPTION_KEY must decode to exactly 32 bytes, got {len(key)}."
        )
    return key


def encrypt_credentials(data: dict[str, Any]) -> str:
    """Serialize and encrypt a credentials dict.

    Returns an opaque versioned string safe for storage in the database.
    """
    plaintext = json.dumps(data, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(_NONCE_SIZE)
    aesgcm = AESGCM(_get_key())
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    b64_nonce = base64.b64encode(nonce).decode()
    b64_ct = base64.b64encode(ciphertext_with_tag).decode()
    return f"{_VERSION}:{b64_nonce}:{b64_ct}"


def decrypt_credentials(blob: str) -> dict[str, Any]:
    """Decrypt and deserialize a credentials blob produced by ``encrypt_credentials``.

    Raises:
        ValueError: If the blob is malformed, tampered, or uses an unknown version.
    """
    parts = blob.split(":", 2)
    if len(parts) != 3:
        raise ValueError("Malformed credentials blob — expected 'version:nonce:ciphertext'.")

    version, b64_nonce, b64_ct = parts
    if version != _VERSION:
        raise ValueError(f"Unsupported credentials blob version '{version}'.")

    try:
        nonce = base64.b64decode(b64_nonce)
        ciphertext_with_tag = base64.b64decode(b64_ct)
    except Exception as exc:
        raise ValueError("Credentials blob contains invalid base64 data.") from exc

    try:
        aesgcm = AESGCM(_get_key())
        plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data=None)
    except Exception as exc:
        raise ValueError(
            "Credentials decryption failed — the blob may be tampered or the key is wrong."
        ) from exc

    try:
        return json.loads(plaintext.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Decrypted credentials are not valid JSON.") from exc
