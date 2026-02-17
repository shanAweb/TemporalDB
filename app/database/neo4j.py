from collections.abc import AsyncGenerator

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from app.config import settings

_driver: AsyncDriver | None = None


def _get_driver() -> AsyncDriver:
    if _driver is None:
        raise RuntimeError("Neo4j driver not initialised. Call init_neo4j() first.")
    return _driver


async def init_neo4j() -> None:
    """Create the Neo4j driver and verify connectivity (called on app startup)."""
    global _driver
    _driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    await _driver.verify_connectivity()


async def close_neo4j() -> None:
    """Close the Neo4j driver (called on app shutdown)."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def get_neo4j() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a Neo4j async session."""
    driver = _get_driver()
    async with driver.session() as session:
        yield session
