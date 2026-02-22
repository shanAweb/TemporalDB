"""
app/api/routes/__init__.py

Shared FastAPI dependencies used across all route modules.
"""
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI dependency that enforces X-API-Key header authentication.

    Returns the validated key so routes can log it if needed.
    Raises HTTP 401 when the key is missing or incorrect.
    """
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
