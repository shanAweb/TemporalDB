"""
app/query/planners/__init__.py

Shared result type returned by every query planner.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.models.sql.event import Event


@dataclass
class PlanResult:
    """Unified result type returned by all query planners.

    Attributes:
        events:       PostgreSQL Event ORM objects retrieved for this query.
        causal_chain: Ordered list of Neo4j chain records.  Each dict has at
                      minimum ``event_id``, ``description``, ``ts_start``,
                      and ``confidence`` keys (as returned by graph_store).
        document_ids: Set of document UUIDs referenced by the returned events,
                      used by the synthesizer to build SourceReference objects.
        confidence:   Planner-level confidence estimate (0.0 – 1.0).
    """

    events: list[Event] = field(default_factory=list)
    causal_chain: list[dict[str, Any]] = field(default_factory=list)
    document_ids: set[uuid.UUID] = field(default_factory=set)
    confidence: float = 1.0
