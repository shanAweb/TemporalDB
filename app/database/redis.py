from collections.abc import AsyncGenerator

from redis.asyncio import ConnectionPool, Redis

from app.config import settings

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("Redis pool not initialised. Call init_redis() first.")
    return _pool


async def init_redis() -> None:
    """Create the Redis connection pool and verify connectivity (called on app startup)."""
    global _pool
    _pool = ConnectionPool.from_url(
        settings.redis_url,
        max_connections=20,
        decode_responses=True,
    )
    # Verify connectivity
    client = Redis(connection_pool=_pool)
    await client.ping()
    await client.aclose()


async def close_redis() -> None:
    """Close the Redis connection pool (called on app shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


async def get_redis() -> AsyncGenerator[Redis, None]:
    """FastAPI dependency that yields a Redis client."""
    pool = _get_pool()
    client = Redis(connection_pool=pool)
    try:
        yield client
    finally:
        await client.aclose()
