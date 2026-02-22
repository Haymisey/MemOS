"""Pydantic request/response models for the MemOS REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class StoreMemoryRequest(BaseModel):
    """Request body for storing a new memory."""

    content: str = Field(..., min_length=1, description="Text content to store")
    source: str = Field("api", description="Source of the memory")
    memory_type: str = Field("note", description="Type/category (note, code, conversation, file)")
    tags: list[str] = Field(default_factory=list, description="Tags for organization")
    metadata: dict | None = Field(None, description="Optional extra metadata")


class SearchRequest(BaseModel):
    """Request body for semantic search."""

    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(5, ge=1, le=100, description="Maximum number of results")
    source: str | None = Field(None, description="Filter by source")
    memory_type: str | None = Field(None, description="Filter by type")
    tags: list[str] | None = Field(None, description="Filter by tags (AND logic)")


class UpdateMemoryRequest(BaseModel):
    """Request body for updating a memory."""

    content: str | None = Field(None, description="New content (re-embeds if changed)")
    source: str | None = None
    memory_type: str | None = None
    tags: list[str] | None = None
    metadata: dict | None = None


class AddEntityRequest(BaseModel):
    """Request body for adding a knowledge graph entity."""

    name: str = Field(..., min_length=1, description="Entity name")
    entity_type: str = Field("concept", description="Entity type (person, project, concept, file)")
    properties: dict | None = Field(None, description="Key-value properties")


class AddRelationshipRequest(BaseModel):
    """Request body for adding a knowledge graph relationship."""

    from_entity: str = Field(..., description="Source entity name")
    to_entity: str = Field(..., description="Target entity name")
    relationship_type: str = Field("related_to", description="Relationship type")
    properties: dict | None = None


class GraphSearchRequest(BaseModel):
    """Request body for searching the knowledge graph."""

    entity_name: str = Field(..., description="Entity to search for")
    relationship_type: str | None = None
    top_k: int = Field(10, ge=1, le=100)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class MemoryResponse(BaseModel):
    """Response model for a single memory."""

    id: str
    content: str
    source: str
    memory_type: str
    tags: list[str]
    created_at: str
    updated_at: str
    metadata: dict = Field(default_factory=dict)


class SearchResultResponse(BaseModel):
    """Response model for a single search result."""

    memory: MemoryResponse
    score: float


class SearchResponse(BaseModel):
    """Response model for search results."""

    results: list[SearchResultResponse]
    count: int
    query: str


class EntityResponse(BaseModel):
    """Response model for a knowledge graph entity."""

    id: str
    name: str
    entity_type: str
    properties: dict = Field(default_factory=dict)
    created_at: str


class RelationshipResponse(BaseModel):
    """Response model for a knowledge graph relationship."""

    id: str
    from_entity: str
    to_entity: str
    relationship_type: str
    properties: dict = Field(default_factory=dict)
    created_at: str


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""

    status: str = "ok"
    version: str
    memory_count: int
    uptime_seconds: float
    backend: str
    embedding_model: str
    embedding_dim: int
    data_dir: str
    connectors: dict[str, Any] = Field(default_factory=dict)


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool
    message: str = ""
