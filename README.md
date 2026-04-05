# TemporalDB

A database that understands **time** and **causality**. Ingest documents, extract events and causal relationships automatically, then query your knowledge base with natural language questions like *"Why did revenue drop in Q3?"* and get structured, cited answers by traversing causal event graphs.

## How It Works

```
Document  ──►  NLP Pipeline  ──►  Event Store (PostgreSQL + pgvector)
                                       │
                                       ▼
                                  Graph Store (Neo4j)
                                       │
                                       ▼
           Natural Language Query  ──►  Query Engine  ──►  Cited Answer
```

1. **Ingest** documents (PDF, DOCX, TXT, Markdown) or text via the REST API.
2. **NLP Pipeline** extracts entities, events, timestamps, and causal relationships automatically.
3. **Dual Storage** persists events in PostgreSQL (with vector embeddings for semantic search) and causal graphs in Neo4j.
4. **Query** with natural language. The engine classifies your intent, traverses the appropriate stores, and synthesizes a cited answer using a local LLM.

## Features

- **Causal Reasoning** — Ask "why" questions and get answers backed by causal chains extracted from your documents.
- **Temporal Awareness** — Query by time ranges, fiscal quarters, relative dates ("last month"), and more.
- **Semantic Search** — Find similar events using vector embeddings (pgvector + all-MiniLM-L6-v2).
- **Entity Resolution** — Automatic deduplication and linking of entities across documents using fuzzy matching and embedding similarity.
- **Structured Citations** — Every answer references source documents, timestamps, and confidence scores.
- **Fully Local** — Runs entirely on your machine. No data leaves your infrastructure. LLM inference via Ollama.

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| Event Store | PostgreSQL 16 + pgvector |
| Graph Store | Neo4j 5 Community Edition |
| Cache / Dedup | Redis 7 |
| Message Bus | Redpanda (Kafka-compatible) |
| NLP | spaCy (NER, dependency parsing) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Local LLM | Ollama (llama3.1:8b for synthesis, codellama:7b for query generation) |
| ORM | SQLAlchemy (async) + asyncpg |
| Migrations | Alembic |
| Task Queue | Celery + Redis |
| Validation | Pydantic v2 |
| Logging | structlog (structured JSON) |
| Containerization | Docker + Docker Compose |

## Project Structure

```
temporaldb-backend/
├── docker-compose.yml              # All infrastructure services
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
├── alembic/                        # Database migrations
│   └── versions/
├── app/
│   ├── main.py                     # FastAPI entry point
│   ├── config.py                   # Settings from environment variables
│   ├── database/
│   │   ├── postgres.py             # Async SQLAlchemy engine + session
│   │   ├── neo4j.py                # Neo4j driver connection
│   │   └── redis.py                # Redis connection
│   ├── models/
│   │   ├── sql/                    # SQLAlchemy ORM models
│   │   │   ├── event.py            # Event table
│   │   │   ├── entity.py           # Entity table
│   │   │   └── document.py         # Source document table
│   │   └── schemas/                # Pydantic request/response schemas
│   │       ├── event.py
│   │       ├── entity.py
│   │       ├── query.py
│   │       └── ingest.py
│   ├── ingestion/
│   │   ├── connectors/
│   │   │   ├── base.py             # Abstract base connector
│   │   │   ├── file.py             # PDF, DOCX, TXT, Markdown
│   │   │   ├── notion.py           # Notion API (planned)
│   │   │   └── confluence.py       # Confluence REST (planned)
│   │   ├── normalizer.py           # Text cleaning and normalization
│   │   ├── deduplicator.py         # SHA-256 fingerprint + Redis check
│   │   └── producer.py             # Kafka producer
│   ├── nlp/
│   │   ├── pipeline.py             # NLP orchestrator
│   │   ├── ner.py                  # Named entity recognition
│   │   ├── coref.py                # Coreference resolution
│   │   ├── event_extractor.py      # Event extraction (SRL + dep parsing)
│   │   ├── temporal_parser.py      # Date/time parsing
│   │   ├── entity_linker.py        # Fuzzy + embedding entity linking
│   │   ├── causal_extractor.py     # Causal relationship extraction
│   │   └── embedder.py             # Embedding generation
│   ├── storage/
│   │   ├── event_store.py          # PostgreSQL CRUD for events
│   │   ├── graph_store.py          # Neo4j CRUD for causal graph
│   │   ├── entity_store.py         # Entity CRUD (both stores)
│   │   └── sync.py                 # PostgreSQL → Neo4j sync
│   ├── query/
│   │   ├── orchestrator.py         # Main query handler
│   │   ├── intent.py               # Intent classification
│   │   ├── temporal_extractor.py   # Time constraint extraction
│   │   ├── entity_resolver.py      # Entity mention → UUID resolution
│   │   ├── planners/
│   │   │   ├── causal_planner.py   # Causal chain traversal
│   │   │   ├── temporal_planner.py # Time range queries
│   │   │   ├── similarity_planner.py # Semantic similarity search
│   │   │   └── entity_planner.py   # Entity timeline queries
│   │   ├── generators/
│   │   │   ├── cypher_gen.py       # LLM-generated Cypher queries
│   │   │   └── sql_gen.py          # LLM-generated SQL queries
│   │   └── synthesizer.py          # Answer synthesis with citations
│   ├── llm/
│   │   ├── client.py               # Ollama client wrapper
│   │   └── prompts.py              # Prompt templates
│   ├── tasks/
│   │   └── nlp_tasks.py            # Celery async tasks
│   └── api/
│       ├── routes/
│       │   ├── query.py            # POST /query
│       │   ├── ingest.py           # POST /ingest, POST /ingest/file
│       │   ├── events.py           # GET /events, GET /events/{id}
│       │   ├── entities.py         # GET /entities, GET /entities/{id}
│       │   └── graph.py            # GET /graph/entity/{id}
│       └── middleware.py           # CORS, rate limiting, logging, auth
├── workers/
│   └── nlp_worker.py               # Kafka consumer entry point
└── tests/
    ├── unit/
    │   ├── test_causal_extractor.py
    │   ├── test_temporal_parser.py
    │   ├── test_intent_classifier.py
    │   └── test_entity_linker.py
    └── integration/
        ├── test_ingest_pipeline.py
        └── test_query_orchestrator.py
```

## Prerequisites

- **Python 3.11+**
- **Docker** and **Docker Compose**
- **Ollama** installed locally with the following models pulled:
  ```bash
  ollama pull llama3.1:8b
  ollama pull codellama:7b
  ```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/temporaldb-backend.git
cd temporaldb-backend
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your preferred settings (defaults work for local development)
```

### 3. Start infrastructure services

```bash
docker compose up -d
```

This starts PostgreSQL (with pgvector), Neo4j, Redis, Redpanda, and pgAdmin.

### 4. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 5. Run database migrations

```bash
alembic upgrade head
```

### 6. Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Start the NLP worker

In a separate terminal:

```bash
python -m workers.nlp_worker
```

### 8. Verify everything is running

```bash
# API health check
curl http://localhost:8000/health

# pgAdmin UI
open http://localhost:5050

# Neo4j Browser
open http://localhost:7474
```

## API Overview

### Ingestion

```bash
# Upload a file
curl -X POST http://localhost:8000/ingest/file \
  -H "X-API-Key: your-api-key" \
  -F "file=@report.pdf"

# Ingest raw text
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Revenue declined 15% in Q3 due to supply chain disruptions.", "source": "quarterly-report"}'
```

### Querying

```bash
# Ask a causal question
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "Why did revenue drop in Q3?"}'

# With filters
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What happened to Acme Corp last quarter?",
    "entity_filter": "Acme Corp",
    "time_range": {"start": "2024-07-01", "end": "2024-09-30"},
    "max_causal_hops": 3
  }'
```

### Response Format

```json
{
  "answer": "Revenue dropped 15% in Q3 primarily due to supply chain disruptions that began in July 2024...",
  "confidence": 0.87,
  "causal_chain": [
    {
      "id": "evt-001",
      "description": "Supply chain disruptions reported",
      "ts_start": "2024-07-15T00:00:00Z",
      "confidence": 0.95
    },
    {
      "id": "evt-002",
      "description": "Production delays across manufacturing",
      "ts_start": "2024-08-01T00:00:00Z",
      "confidence": 0.88
    },
    {
      "id": "evt-003",
      "description": "Revenue declined 15% in Q3",
      "ts_start": "2024-10-01T00:00:00Z",
      "confidence": 0.92
    }
  ],
  "sources": [
    {
      "id": "doc-001",
      "source": "quarterly-report",
      "metadata": {"filename": "Q3-2024-Report.pdf"}
    }
  ]
}
```

### Browse Data

```bash
# List events
curl http://localhost:8000/events?entity_id=uuid&from_date=2024-01-01&limit=20

# Get single event
curl http://localhost:8000/events/{event_id}

# Search entities
curl http://localhost:8000/entities?name=Acme

# Get entity causal graph
curl http://localhost:8000/graph/entity/{entity_id}
```

## Query Types

| Intent | Example Question | Engine |
|---|---|---|
| **CAUSAL_WHY** | "Why did revenue drop in Q3?" | Neo4j causal graph traversal |
| **TEMPORAL_RANGE** | "What happened between July and September?" | PostgreSQL range scan |
| **SIMILARITY** | "Find events similar to the supply chain disruption" | pgvector cosine similarity |
| **ENTITY_TIMELINE** | "Show me everything about Acme Corp" | Combined PostgreSQL + Neo4j |

## Architecture

### Data Flow

```
                    ┌─────────────┐
                    │  REST API   │
                    │  (FastAPI)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Ingest  │ │  Query   │ │  Browse  │
        │ Endpoint │ │ Endpoint │ │ Endpoints│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
             ▼             │             │
        ┌──────────┐       │             │
        │ Redpanda │       │             │
        │ (Kafka)  │       │             │
        └────┬─────┘       │             │
             │             │             │
             ▼             │             │
        ┌──────────┐       │             │
        │   NLP    │       │             │
        │ Pipeline │       │             │
        └────┬─────┘       │             │
             │             │             │
             ▼             ▼             ▼
        ┌─────────────────────────────────────┐
        │           Storage Layer             │
        │  ┌────────────┐  ┌──────────────┐   │
        │  │ PostgreSQL │  │    Neo4j     │   │
        │  │ + pgvector │◄─┤ Causal Graph │   │
        │  └────────────┘  └──────────────┘   │
        └─────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Ollama    │
                    │  (Local LLM)│
                    └─────────────┘
```

### NLP Pipeline Stages

1. **Named Entity Recognition** — spaCy identifies people, organizations, dates, etc.
2. **Coreference Resolution** — Pronouns and references linked to canonical entities.
3. **Event Extraction** — Subject-verb-object tuples extracted from each sentence.
4. **Temporal Parsing** — Date expressions normalized to UTC timestamps.
5. **Entity Linking** — Mentions matched to canonical entities (exact → fuzzy → embedding).
6. **Causal Extraction** — Causal cue phrases identify cause-effect relationships.
7. **Embedding Generation** — Dense vector representations for semantic search.

## Development

### Running Tests

```bash
# Unit tests (no infrastructure required)
pytest tests/unit/ -v

# Integration tests (requires Docker services running)
pytest tests/integration/ -v

# All tests
pytest -v
```

### Service Ports

| Service | Port | UI |
|---|---|---|
| FastAPI | 8000 | http://localhost:8000/docs |
| PostgreSQL | 5432 | — |
| Neo4j | 7474 / 7687 | http://localhost:7474 |
| Redis | 6379 | — |
| Redpanda | 9092 | — |
| pgAdmin | 5050 | http://localhost:5050 |

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Follow the coding standards:
   - All async functions in API routes.
   - Type hints on all function signatures, including return types.
   - Docstrings on every function.
   - Configuration from `app/config.py` only — no hardcoded values.
   - Parameterized database queries — no string interpolation in SQL/Cypher.
   - Structured logging via `structlog`.
   - Consistent error responses: `{"error": str, "detail": str, "code": str}`.
4. Write tests for new functionality.
5. Submit a pull request.

## License

MIT
