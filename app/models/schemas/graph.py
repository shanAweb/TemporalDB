from uuid import UUID

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A node in the causal graph."""
    id: UUID
    label: str
    type: str = Field(..., description="event or entity")
    properties: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """An edge in the causal graph."""
    source: UUID
    target: UUID
    type: str = Field(..., description="CAUSED_BY, INVOLVES, etc.")
    confidence: float | None = None


class GraphResponse(BaseModel):
    """Subgraph response for graph visualization."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
