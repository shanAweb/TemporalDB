# =============================================================================
# TemporalDB Backend — Dockerfile
# =============================================================================
# Multi-stage build:
#   builder  — installs Python packages and pre-caches NLP models
#   runtime  — lean production image (no build tools)
#
# Build arguments (override with --build-arg):
#   SPACY_MODEL      spaCy pipeline to download   (default: en_core_web_trf)
#   EMBEDDING_MODEL  HuggingFace model to cache   (default: all-MiniLM-L6-v2)
#
# To run the NLP worker instead of the API server, override CMD:
#   docker run ... temporaldb python -m workers.nlp_worker
# =============================================================================

# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install production Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-prod.txt

# Download spaCy model (installs as a Python package under site-packages)
ARG SPACY_MODEL=en_core_web_trf
RUN python -m spacy download ${SPACY_MODEL}

# Pre-cache the sentence-transformers / HuggingFace embedding model
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN HF_HOME=/tmp/hf_cache \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"


# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r appuser \
 && useradd --no-log-init -r -g appuser -d /home/appuser -m appuser

# Copy installed Python packages and executables from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy HuggingFace model cache into the user home directory
COPY --from=builder --chown=appuser:appuser \
     /tmp/hf_cache /home/appuser/.cache/huggingface

# Application directory
WORKDIR /app
COPY --chown=appuser:appuser app/       ./app/
COPY --chown=appuser:appuser alembic/   ./alembic/
COPY --chown=appuser:appuser alembic.ini .

# Python runtime flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--no-access-log"]
