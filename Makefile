# =============================================================================
# TemporalDB Backend — Makefile
# =============================================================================
# Usage:
#   make <target>
#   make help            list all targets with descriptions
#
# Variables (override on the command line):
#   PYTHON          Python interpreter          (default: python)
#   PORT            API server port             (default: 8000)
#   SPACY_MODEL     spaCy model to download     (default: en_core_web_trf)
#   IMAGE_NAME      Docker image name           (default: temporaldb)
#   IMAGE_TAG       Docker image tag            (default: latest)
# =============================================================================

PYTHON      ?= python
PORT        ?= 8000
SPACY_MODEL ?= en_core_web_trf
IMAGE_NAME  ?= temporaldb
IMAGE_TAG   ?= latest

.DEFAULT_GOAL := help

.PHONY: help \
        install install-prod setup spacy-download \
        up down down-v logs ps db-shell \
        migrate migrate-down migrate-status migrate-history migration \
        run worker \
        test test-unit test-api test-integration test-cov test-all \
        lint fmt fmt-check check \
        docker-build docker-run \
        clean


# ─── Help ──────────────────────────────────────────────────────────────────

help:  ## Show this help message
	@echo "TemporalDB Backend"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""


# ─── Setup / Install ──────────────────────────────────────────────────────

install:  ## Install all dependencies including dev/test tools
	pip install -r requirements.txt

install-prod:  ## Install production dependencies only
	pip install -r requirements-prod.txt

setup:  ## Bootstrap local environment: copy .env.example and install deps
	@if [ ! -f .env ]; then \
	    cp .env.example .env; \
	    echo "Created .env from .env.example — update values before starting."; \
	fi
	$(PYTHON) -m pip install -r requirements.txt

spacy-download:  ## Download the spaCy model (override: make spacy-download SPACY_MODEL=en_core_web_sm)
	$(PYTHON) -m spacy download $(SPACY_MODEL)


# ─── Infrastructure (Docker Compose) ──────────────────────────────────────

up:  ## Start all backing services (PostgreSQL, Neo4j, Redis, Redpanda)
	docker compose up -d

down:  ## Stop all backing services
	docker compose down

down-v:  ## Stop all backing services and delete volumes  [destructive]
	docker compose down -v

logs:  ## Tail logs from all running services
	docker compose logs -f

ps:  ## Show status of all services
	docker compose ps

db-shell:  ## Open a psql shell inside the running postgres container
	docker compose exec postgres psql -U temporaldb temporaldb


# ─── Database / Migrations ────────────────────────────────────────────────

migrate:  ## Apply all pending Alembic migrations (upgrade head)
	alembic upgrade head

migrate-down:  ## Roll back the last Alembic migration
	alembic downgrade -1

migrate-status:  ## Show the current applied Alembic revision
	alembic current

migrate-history:  ## Show full Alembic migration history
	alembic history --verbose

migration:  ## Auto-generate a new migration: make migration msg="describe change"
	@[ -n "$(msg)" ] || \
	    (echo "Error: provide a message — make migration msg=\"your message\"" && exit 1)
	alembic revision --autogenerate -m "$(msg)"


# ─── Development Server ───────────────────────────────────────────────────

run:  ## Start the API server with hot-reload (development)
	uvicorn app.main:app \
	    --host 0.0.0.0 \
	    --port $(PORT) \
	    --reload \
	    --log-level info

worker:  ## Start the NLP Kafka consumer worker
	$(PYTHON) -m workers.nlp_worker


# ─── Tests ────────────────────────────────────────────────────────────────

test:  ## Run unit and API tests
	pytest tests/unit tests/api -v

test-unit:  ## Run unit tests only
	pytest tests/unit -v

test-api:  ## Run API endpoint tests only
	pytest tests/api -v

test-integration:  ## Run integration tests (requires running Docker services)
	pytest tests/integration -v

test-cov:  ## Run unit + API tests with terminal and HTML coverage report
	pytest tests/unit tests/api \
	    --cov=app \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    -v

test-all:  ## Run the full test suite including integration tests
	pytest tests/ -v


# ─── Code Quality ─────────────────────────────────────────────────────────

lint:  ## Check code style with ruff
	ruff check .

fmt:  ## Auto-format code with ruff
	ruff format .

fmt-check:  ## Check formatting without making changes
	ruff format --check .

check:  ## Run lint and format checks together (CI-safe, no writes)
	ruff check . && ruff format --check .


# ─── Docker ───────────────────────────────────────────────────────────────

docker-build:  ## Build the production Docker image
	docker build \
	    --build-arg SPACY_MODEL=en_core_web_trf \
	    --build-arg EMBEDDING_MODEL=all-MiniLM-L6-v2 \
	    -t $(IMAGE_NAME):$(IMAGE_TAG) \
	    .

docker-run:  ## Run the API server container locally (requires .env)
	docker run --rm \
	    --env-file .env \
	    -p $(PORT):8000 \
	    $(IMAGE_NAME):$(IMAGE_TAG)


# ─── Cleanup ──────────────────────────────────────────────────────────────

clean:  ## Remove bytecode, test caches, and coverage artifacts
	find . -type d -name __pycache__ -not -path "./temporalenv/*" | xargs rm -rf
	find . -type f -name "*.pyc"     -not -path "./temporalenv/*" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
