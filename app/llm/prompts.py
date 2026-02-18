"""Prompt templates for all Ollama LLM interactions."""

# ── Intent Classification ────────────────────────────────

INTENT_CLASSIFICATION = """You are an intent classifier for a temporal-causal database.

Classify the following user query into exactly ONE of these intents:
- CAUSAL_WHY: Questions asking "why" something happened, seeking cause-effect chains
- TEMPORAL_RANGE: Questions about what happened during a specific time period
- SIMILARITY: Questions asking for events similar to a described event
- ENTITY_TIMELINE: Questions about the history or timeline of a specific entity

User query: {query}

Respond with ONLY the intent label, nothing else."""


# ── Event Extraction ─────────────────────────────────────

EVENT_EXTRACTION = """You are an event extraction system. Extract structured events from the given text.

For each event, identify:
- description: A concise description of the event
- subjects: The entities involved (people, organizations, etc.)
- temporal_expression: Any date, time, or temporal phrase associated with the event
- event_type: One of [action, state_change, declaration, occurrence]

Text:
{text}

Return a JSON array of events. Each event should be a JSON object with the keys: description, subjects, temporal_expression, event_type.
Respond with ONLY valid JSON, no other text."""


# ── Causal Relationship Extraction ───────────────────────

CAUSAL_EXTRACTION = """You are a causal relationship extractor. Analyze the given text and identify cause-effect relationships between events.

For each causal relationship, identify:
- cause: Description of the cause event
- effect: Description of the effect event
- confidence: Your confidence in this causal link (0.0 to 1.0)
- evidence: The phrase or sentence that indicates this causal relationship

Text:
{text}

Return a JSON array of causal relationships. Each should be a JSON object with keys: cause, effect, confidence, evidence.
Respond with ONLY valid JSON, no other text."""


# ── Cypher Query Generation ──────────────────────────────

CYPHER_GENERATION = """You are a Neo4j Cypher query generator for a causal event graph.

Graph schema:
- Nodes: (:Event {{id, description, ts_start, ts_end, confidence}})
- Nodes: (:Entity {{id, name, type}})
- Relationships: [:CAUSED_BY {{confidence, evidence}}] (Event)-[:CAUSED_BY]->(Event)
- Relationships: [:INVOLVES] (Event)-[:INVOLVES]->(Entity)

User intent: {intent}
Query parameters: {parameters}

Generate a Cypher query that answers the user's question. Use parameterised values with $param syntax where appropriate.
Respond with ONLY the Cypher query, no explanation."""


# ── SQL Query Generation ─────────────────────────────────

SQL_GENERATION = """You are a PostgreSQL query generator for a temporal event database.

Table schema:
- events (id UUID, description TEXT, ts_start TIMESTAMPTZ, ts_end TIMESTAMPTZ, confidence FLOAT, document_id UUID, embedding VECTOR(384))
- entities (id UUID, name TEXT, type VARCHAR, canonical_name TEXT)
- event_entities (event_id UUID, entity_id UUID)
- documents (id UUID, source TEXT, filename TEXT, ingested_at TIMESTAMPTZ)

User intent: {intent}
Query parameters: {parameters}

Generate a PostgreSQL query that answers the user's question. Use parameterised values with :param syntax where appropriate.
Respond with ONLY the SQL query, no explanation."""


# ── Answer Synthesis ─────────────────────────────────────

ANSWER_SYNTHESIS = """You are a research assistant that synthesizes answers from structured data.

Given the following query results, compose a clear, concise answer to the user's question. Include:
1. A direct answer to the question
2. Supporting evidence from the causal chain (if available)
3. Reference source documents by their source name

User question: {question}

Retrieved events:
{events}

Causal chain (if any):
{causal_chain}

Source documents:
{sources}

Provide a well-structured answer with citations. Be factual — only state what the data supports."""


# ── Coreference Resolution ───────────────────────────────

COREF_RESOLUTION = """You are a coreference resolution system. Given a text, replace all pronouns and ambiguous references with the explicit entity they refer to.

Text:
{text}

Return the rewritten text with all pronouns and references replaced by their explicit entity names. Do not change anything else about the text."""
