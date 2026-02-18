from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────
    app_name: str = "TemporalDB"
    app_version: str = "0.1.0"
    debug: bool = False
    api_key: str = "changeme"

    # ── PostgreSQL ───────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "temporaldb"
    postgres_password: str = "temporaldb"
    postgres_db: str = "temporaldb"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        """Sync DSN for Alembic migrations."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Neo4j ────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7688"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "temporaldb"

    # ── Redis ────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6380
    redis_db: int = 0
    redis_password: str = ""

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── Kafka / Redpanda ─────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:29092"
    kafka_ingestion_topic: str = "document.ingested"
    kafka_consumer_group: str = "temporaldb-nlp"

    # ── Ollama ───────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_synthesis_model: str = "llama3.1:8b"
    ollama_codegen_model: str = "codellama:7b"
    ollama_request_timeout: float = 120.0

    # ── NLP ──────────────────────────────────────────────
    spacy_model: str = "en_core_web_trf"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Celery ───────────────────────────────────────────
    @property
    def celery_broker_url(self) -> str:
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        return self.redis_url

    # ── Server ───────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
