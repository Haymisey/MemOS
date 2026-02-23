"""MemoryEngine — the high-level orchestrator for MemOS.

This is the main entry point for all memory operations. It composes the
EmbeddingEngine and a VectorStoreBackend to provide a simple, high-level API:

    engine = MemoryEngine(config)
    engine.initialize()
    engine.store("I just learned about Zvec", source="cursor", tags=["coding"])
    results = engine.search("vector databases")
"""

from __future__ import annotations

import logging
import time
from typing import Any

from memos.core.base import Memory, SearchFilter, SearchResult, VectorStoreBackend, UNDEFINED, generate_id
from memos.core.config import MemOSConfig
from memos.core.embeddings import EmbeddingEngine

logger = logging.getLogger("memos.core.memory_engine")


class MemoryEngine:
    """High-level memory management engine.

    Coordinates embedding generation and vector storage. This class is
    backend-agnostic — it only talks to the VectorStoreBackend ABC.

    Args:
        config:  MemOS configuration.
        backend: An optional pre-configured backend. If None, one is
                 created automatically based on config.backend.
    """

    def __init__(
        self,
        config: MemOSConfig | None = None,
        backend: VectorStoreBackend | None = None,
    ) -> None:
        self.config = config or MemOSConfig()
        self._embedder = EmbeddingEngine(model_name=self.config.embedding_model)
        self._backend = backend
        self._start_time = time.time()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the engine: set up storage backend and embedding model.

        Must be called before any other operations.
        """
        if self._initialized:
            return

        self.config.ensure_dirs()

        # Create backend from config if not provided
        if self._backend is None:
            self._backend = self._create_backend()

        self._backend.initialize()
        self._initialized = True
        self._start_time = time.time()
        logger.info(
            "MemoryEngine initialized (backend=%s, data_dir=%s)",
            self.config.backend,
            self.config.data_dir,
        )

    def _create_backend(self) -> VectorStoreBackend:
        """Factory method to create a backend based on config."""
        if self.config.backend == "zvec":
            from memos.core.zvec_backend import ZvecBackend

            return ZvecBackend(
                db_path=self.config.db_path,
                embedding_dim=self.config.embedding_dim,
            )
        elif self.config.backend == "lancedb":
            from memos.core.lancedb_backend import LanceDBBackend

            return LanceDBBackend(
                db_path=self.config.db_path,
                embedding_dim=self.config.embedding_dim,
            )
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def _ensure_initialized(self) -> VectorStoreBackend:
        """Ensure the engine is initialized, return the backend."""
        if not self._initialized or self._backend is None:
            raise RuntimeError("MemoryEngine not initialized. Call initialize() first.")
        return self._backend

    # -------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------

    def store(
        self,
        content: str,
        source: str = "unknown",
        memory_type: str = "note",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_hours: int | None = None,
    ) -> Memory:
        """Store a new memory.

        The content is automatically embedded into a vector before storage.

        Args:
            content:     The text content to store.
            source:      Where this came from (e.g., "cli", "clipboard", "cursor").
            memory_type: Category (e.g., "note", "code", "conversation").
            tags:        Optional list of tags for organization.
            metadata:    Optional extra metadata dict.
            ttl_hours:   Optional override for Time-To-Live in hours.
                         If None, "clipboard" source gets default TTL from config.

        Returns:
            The stored Memory object with its generated ID and embedding.
        """
        backend = self._ensure_initialized()
        from datetime import datetime, timezone, timedelta

        # Handle auto-expiry for specific sources
        expires_at = None
        if ttl_hours is not None:
            expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
        elif source == "clipboard":
            expires_at = (datetime.now(timezone.utc) + timedelta(hours=self.config.clipboard_ttl_hours)).isoformat()

        # Generate embedding
        embedding = self._embedder.embed_text(content)

        # Generate unique ID
        memory_id = generate_id()

        # Store in backend
        backend.add(
            id=memory_id,
            content=content,
            embedding=embedding,
            source=source,
            memory_type=memory_type,
            tags=tags or [],
            metadata=metadata,
            expires_at=expires_at,
            is_pinned=False,
        )

        logger.info("Stored memory %s (source=%s, %d chars, expires=%s)", 
                    memory_id, source, len(content), expires_at)

        # Fetch and return the complete memory
        memory = backend.get(memory_id)
        if memory is None:
            # Construct manually if fetch fails (shouldn't happen)
            now = datetime.now(timezone.utc).isoformat()
            memory = Memory(
                id=memory_id,
                content=content,
                embedding=embedding,
                source=source,
                memory_type=memory_type,
                tags=tags or [],
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                is_pinned=False,
                metadata=metadata or {},
            )

        return memory

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilter | None = None,
    ) -> list[SearchResult]:
        """Semantic search for memories related to a query.

        The query is automatically embedded and compared against stored memories.

        Args:
            query:   Natural language search query.
            top_k:   Maximum number of results (default: 5).
            filters: Optional filters to narrow results.

        Returns:
            List of SearchResult, sorted by descending relevance.
        """
        backend = self._ensure_initialized()

        query_embedding = self._embedder.embed_text(query)
        results = backend.search(query_embedding, top_k=top_k, filters=filters)

        logger.debug("Search for '%s' returned %d results", query[:50], len(results))
        return results

    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The unique memory identifier.

        Returns:
            The Memory if found, else None.
        """
        backend = self._ensure_initialized()
        return backend.get(memory_id)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        source: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = UNDEFINED,
        is_pinned: bool | None = UNDEFINED,
    ) -> bool:
        """Update an existing memory.

        If content is changed, a new embedding is automatically generated.

        Args:
            memory_id:   The memory to update.
            content:     New content (re-embeds if changed).
            source:      New source.
            memory_type: New type.
            tags:        New tags.
            metadata:    New metadata.

        Returns:
            True if updated, False if memory not found.
        """
        backend = self._ensure_initialized()

        # Re-embed if content changed
        new_embedding = None
        if content is not None:
            new_embedding = self._embedder.embed_text(content)

        success = backend.update(
            id=memory_id,
            content=content,
            embedding=new_embedding,
            source=source,
            memory_type=memory_type,
            tags=tags,
            metadata=metadata,
            expires_at=expires_at,
            is_pinned=is_pinned,
        )

        if success:
            logger.info("Updated memory %s", memory_id)
        return success

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        backend = self._ensure_initialized()
        success = backend.delete(memory_id)
        if success:
            logger.info("Deleted memory %s", memory_id)
        return success

    def list_memories(self, filters: SearchFilter | None = None) -> list[Memory]:
        """List all stored memories, optionally filtered.

        Args:
            filters: Optional filters.

        Returns:
            List of Memory objects.
        """
        backend = self._ensure_initialized()
        return backend.list_all(filters=filters)

    def cleanup_expired(self) -> int:
        """Find and delete expired memories. Delegate to backend."""
        backend = self._ensure_initialized()
        count = backend.cleanup_expired()
        if count > 0:
            logger.info("Garbage collection: purged %d memories", count)
        return count

    def pin(self, memory_id: str) -> bool:
        """Mark a memory as pinned (permanent) and remove its expiry."""
        return self.update(memory_id, is_pinned=True, expires_at=None)

    def unpin(self, memory_id: str, ttl_hours: int | UNDEFINED = UNDEFINED) -> bool:
        """Unpin a memory, making it eligible for auto-expiry again."""
        from datetime import datetime, timezone, timedelta
        expires_at = UNDEFINED
        if ttl_hours is not UNDEFINED:
            if ttl_hours is not None:
                expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
            else:
                expires_at = None
        
        return self.update(memory_id, is_pinned=False, expires_at=expires_at)

    # -------------------------------------------------------------------
    # Status & Lifecycle
    # -------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Get daemon/engine statistics.

        Returns:
            Dict with count, uptime, backend info, etc.
        """
        backend = self._ensure_initialized()
        uptime = time.time() - self._start_time

        return {
            "memory_count": backend.count(),
            "uptime_seconds": round(uptime, 1),
            "backend": self.config.backend,
            "embedding_model": self.config.embedding_model,
            "embedding_dim": self.config.embedding_dim,
            "data_dir": str(self.config.data_dir),
        }

    def close(self) -> None:
        """Gracefully shut down the engine."""
        if self._backend is not None:
            self._backend.close()
            self._initialized = False
            logger.info("MemoryEngine shut down.")
