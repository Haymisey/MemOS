"""Abstract base class for vector store backends and shared data models.

This module defines the contract that ALL vector store backends must implement.
Swap Zvec → LanceDB → ChromaDB without touching any other code.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Memory:
    """A single memory stored in the vector database.

    Attributes:
        id:          Unique identifier (UUID string).
        content:     The raw text content of the memory.
        embedding:   The vector embedding (list of floats).
        source:      Where this memory came from (e.g., "cli", "clipboard", "cursor").
        memory_type: Category of memory (e.g., "note", "code", "conversation", "file").
        tags:        User-defined tags for organization.
        created_at:  ISO timestamp of when the memory was created.
        updated_at:  ISO timestamp of the last update.
        metadata:    Optional extra metadata dict.
    """

    id: str
    content: str
    embedding: list[float] = field(default_factory=list, repr=False)
    source: str = "unknown"
    memory_type: str = "note"
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single result from a semantic search query.

    Attributes:
        memory: The matched Memory object.
        score:  Similarity score (higher = more relevant).
    """

    memory: Memory
    score: float


@dataclass
class SearchFilter:
    """Filters to narrow down search or list results.

    All fields are optional. When multiple are set, they are AND-combined.

    Attributes:
        source:       Filter by source (e.g., "clipboard").
        memory_type:  Filter by memory type (e.g., "code").
        tags:         Filter by tags (memory must have ALL specified tags).
        created_after:  Only return memories created after this ISO timestamp.
        created_before: Only return memories created before this ISO timestamp.
    """

    source: str | None = None
    memory_type: str | None = None
    tags: list[str] | None = None
    created_after: str | None = None
    created_before: str | None = None


def generate_id() -> str:
    """Generate a unique memory ID."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class VectorStoreBackend(ABC):
    """Abstract interface for vector store backends.

    Any backend (Zvec, LanceDB, ChromaDB, etc.) must implement these methods.
    The MemoryEngine orchestrator uses this interface exclusively — it never
    talks to any backend directly.

    Lifecycle:
        1. Construct the backend instance
        2. Call `initialize()` to set up storage
        3. Use `add`, `search`, `get`, `update`, `delete`, `list_all`
        4. Call `close()` when shutting down
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend: create collections, ensure schema, etc.

        Must be called before any other operations.
        """

    @abstractmethod
    def add(
        self,
        id: str,
        content: str,
        embedding: list[float],
        source: str,
        memory_type: str,
        tags: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new memory.

        Args:
            id:          Unique ID for the memory.
            content:     Raw text content.
            embedding:   Vector embedding.
            source:      Source identifier.
            memory_type: Type/category.
            tags:        List of tags.
            metadata:    Optional extra metadata.

        Returns:
            The ID of the stored memory.
        """

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: SearchFilter | None = None,
    ) -> list[SearchResult]:
        """Semantic search for memories similar to the given embedding.

        Args:
            embedding: Query vector.
            top_k:     Maximum number of results to return.
            filters:   Optional filters to narrow results.

        Returns:
            List of SearchResult, sorted by descending similarity score.
        """

    @abstractmethod
    def get(self, id: str) -> Memory | None:
        """Retrieve a single memory by its ID.

        Returns:
            The Memory if found, else None.
        """

    @abstractmethod
    def update(
        self,
        id: str,
        content: str | None = None,
        embedding: list[float] | None = None,
        source: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing memory's fields.

        Only non-None fields are updated. The `updated_at` timestamp is
        always refreshed.

        Returns:
            True if the memory was found and updated, False otherwise.
        """

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a memory by its ID.

        Returns:
            True if the memory was found and deleted, False otherwise.
        """

    @abstractmethod
    def list_all(self, filters: SearchFilter | None = None) -> list[Memory]:
        """List all memories, optionally filtered.

        Args:
            filters: Optional filters to narrow results.

        Returns:
            List of Memory objects matching the filters.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored memories."""

    @abstractmethod
    def close(self) -> None:
        """Gracefully shut down the backend and release resources."""
