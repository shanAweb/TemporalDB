from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database.postgres import init_postgres, close_postgres
from app.database.neo4j import init_neo4j, close_neo4j
from app.database.redis import init_redis, close_redis

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown of all service connections."""
    logger.info("starting_up", app=settings.app_name, version=settings.app_version)

    # ── Startup ──────────────────────────────────────────
    await init_postgres()
    logger.info("postgres_connected")

    await init_neo4j()
    logger.info("neo4j_connected")

    await init_redis()
    logger.info("redis_connected")

    logger.info("startup_complete")
    yield

    # ── Shutdown ─────────────────────────────────────────
    logger.info("shutting_down")
    await close_redis()
    await close_neo4j()
    await close_postgres()
    logger.info("shutdown_complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A database that understands time and causality.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ───────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred.",
            "code": "INTERNAL_ERROR",
        },
    )


# ── Health Check ─────────────────────────────────────────

@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


# ── Routers ──────────────────────────────────────────────
# Will be added as API routes are built in Phase 7:
# from app.api.routes import query, ingest, events, entities, graph
# app.include_router(query.router, prefix="/query", tags=["Query"])
# app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
# app.include_router(events.router, prefix="/events", tags=["Events"])
# app.include_router(entities.router, prefix="/entities", tags=["Entities"])
# app.include_router(graph.router, prefix="/graph", tags=["Graph"])
