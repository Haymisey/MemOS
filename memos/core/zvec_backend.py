"""Zvec backend implementation for the VectorStoreBackend ABC.

This is the DEFAULT backend for MemOS — Alibaba's embedded vector database,
the "SQLite of vector databases". Zero infrastructure, in-process, blazing fast.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memos.core.base import Memory, SearchFilter, SearchResult, VectorStoreBackend

logger = logging.getLogger("memos.core.zvec_backend")

class ZvecBackend(VectorStoreBackend):
    """Zvec implementation of the VectorStoreBackend interface."""

    def __init__(self, db_path: Path, embedding_dim: int = 384) -> None:
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._collection: Any = None

    def initialize(self) -> None:
        """Create or open the memories collection."""
        import zvec
        self._db_path.mkdir(parents=True, exist_ok=True)
        collection_path = str(self._db_path / "memories")

        schema = zvec.CollectionSchema(
            name="memories",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, self._embedding_dim),
            fields=[
                zvec.ScalarSchema("content", zvec.DataType.STRING),
                zvec.ScalarSchema("source", zvec.DataType.STRING),
                zvec.ScalarSchema("memory_type", zvec.DataType.STRING),
                zvec.ScalarSchema("tags_json", zvec.DataType.STRING),
                zvec.ScalarSchema("created_at", zvec.DataType.STRING),
                zvec.ScalarSchema("updated_at", zvec.DataType.STRING),
                zvec.ScalarSchema("metadata_json", zvec.DataType.STRING),
            ],
        )

        try:
            self._collection = zvec.open(path=collection_path)
            logger.info("Opened existing Zvec collection at %s", collection_path)
        except Exception:
            self._collection = zvec.create_and_open(path=collection_path, schema=schema)
            logger.info("Created new Zvec collection at %s", collection_path)

    def _ensure_collection(self) -> Any:
        """Return the active collection, raising if not initialized."""
        if self._collection is None:
            raise RuntimeError("ZvecBackend not initialized. Call initialize() first.")
        return self._collection

    def add(
        self,
        id: str,
        content: str,
        embedding: list[float],
        source: str = "unknown",
        memory_type: str = "note",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new memory in Zvec."""
        collection = self._ensure_collection()
        now = datetime.now(timezone.utc).isoformat()

        doc = zvec.Doc(
            id=id,
            vectors={"embedding": embedding},
            fields={
                "content": content,
                "source": source,
                "memory_type": memory_type,
                "tags_json": json.dumps(tags or []),
                "created_at": now,
                "updated_at": now,
                "metadata_json": json.dumps(metadata or {}),
            },
        )

        result = collection.insert(doc)
        if hasattr(result, "code") and result.code != 0:
            raise RuntimeError(f"Zvec insert failed: {result}")

        logger.debug("Stored memory %s (source=%s, type=%s)", id, source, memory_type)
        return id

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: SearchFilter | None = None,
    ) -> list[SearchResult]:
        """Semantic search using vector similarity."""
        collection = self._ensure_collection()

        # Build Zvec filter string for supported scalar filters
        filter_str = self._build_filter_string(filters)

        query_kwargs: dict[str, Any] = {
            "vectors": zvec.VectorQuery(field_name="embedding", vector=embedding),
            "topk": top_k * 3 if filters and filters.tags else top_k,  # Over-fetch if tag filtering needed
        }
        if filter_str:
            query_kwargs["filter"] = filter_str

        results = collection.query(**query_kwargs)

        # Convert Zvec results to SearchResult objects
        search_results: list[SearchResult] = []
        for hit in results:
            memory = self._hit_to_memory(hit)
            if memory is None:
                continue

            # Post-filter by tags (Zvec doesn't support array filtering)
            if filters and filters.tags:
                if not all(t in memory.tags for t in filters.tags):
                    continue

            search_results.append(SearchResult(memory=memory, score=hit.score))

            if len(search_results) >= top_k:
                break

        return search_results

    def get(self, id: str) -> Memory | None:
        """Fetch a single memory by ID."""
        collection = self._ensure_collection()

        try:
            result = collection.fetch(ids=id)
            if result and len(result) > 0:
                return self._doc_to_memory(result[0])
        except Exception as e:
            logger.debug("Failed to fetch memory %s: %s", id, e)

        return None

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
        """Update an existing memory using upsert."""
        existing = self.get(id)
        if existing is None:
            return False

        collection = self._ensure_collection()
        now = datetime.now(timezone.utc).isoformat()

        doc = zvec.Doc(
            id=id,
            vectors={"embedding": embedding or existing.embedding},
            fields={
                "content": content or existing.content,
                "source": source or existing.source,
                "memory_type": memory_type or existing.memory_type,
                "tags_json": json.dumps(tags if tags is not None else existing.tags),
                "created_at": existing.created_at,
                "updated_at": now,
                "metadata_json": json.dumps(
                    metadata if metadata is not None else existing.metadata
                ),
            },
        )

        result = collection.upsert(doc)
        if hasattr(result, "code") and result.code != 0:
            logger.warning("Zvec upsert failed for %s: %s", id, result)
            return False

        logger.debug("Updated memory %s", id)
        return True

    def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        collection = self._ensure_collection()
        try:
            collection.delete(ids=id)
            logger.debug("Deleted memory %s", id)
            return True
        except Exception as e:
            logger.warning("Failed to delete memory %s: %s", id, e)
            return False

    def list_all(self, filters: SearchFilter | None = None) -> list[Memory]:
        """List all memories, optionally filtered.

        Since Zvec doesn't have a native 'list all' without a query vector,
        we use a conditional filter query or fetch approach.
        """
        collection = self._ensure_collection()

        filter_str = self._build_filter_string(filters)

        try:
            # Query with filter only (no vector) if filter exists,
            # otherwise fetch a large batch
            if filter_str:
                results = collection.query(filter=filter_str, topk=10000)
            else:
                results = collection.query(topk=10000)

            memories: list[Memory] = []
            for hit in results:
                memory = self._hit_to_memory(hit)
                if memory is None:
                    continue

                # Post-filter by tags
                if filters and filters.tags:
                    if not all(t in memory.tags for t in filters.tags):
                        continue

                memories.append(memory)

            return memories
        except Exception as e:
            logger.warning("list_all failed: %s", e)
            return []

    def count(self) -> int:
        """Return total number of stored memories."""
        collection = self._ensure_collection()
        try:
            stats = collection.stats
            # Zvec stats object has doc_count or similar attribute
            if hasattr(stats, "doc_count"):
                return stats.doc_count
            if hasattr(stats, "total_doc_count"):
                return stats.total_doc_count
            # Try string parsing as fallback
            stats_str = str(stats)
            logger.debug("Collection stats: %s", stats_str)
            return 0
        except Exception as e:
            logger.debug("count() failed: %s", e)
            return 0

    def close(self) -> None:
        """Close the Zvec collection and release resources."""
        if self._collection is not None:
            try:
                self._collection.close()
            except Exception:
                pass
            self._collection = None
            logger.info("Zvec backend closed.")

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _build_filter_string(self, filters: SearchFilter | None) -> str | None:
        """Convert a SearchFilter to a Zvec filter expression string."""
        if filters is None:
            return None

        conditions: list[str] = []

        if filters.source:
            conditions.append(f'source = "{filters.source}"')
        if filters.memory_type:
            conditions.append(f'memory_type = "{filters.memory_type}"')
        if filters.created_after:
            conditions.append(f'created_at > "{filters.created_after}"')
        if filters.created_before:
            conditions.append(f'created_at < "{filters.created_before}"')

        # Tags are filtered in Python (post-query) since Zvec doesn't
        # support JSON array contains.

        if not conditions:
            return None

        return " AND ".join(conditions)

    def _hit_to_memory(self, hit: Any) -> Memory | None:
        """Convert a Zvec query hit to a Memory object."""
        try:
            fields = hit.fields if hasattr(hit, "fields") else {}
            doc_id = hit.id if hasattr(hit, "id") else str(hit.get("id", ""))

            # Extract the embedding vector if available
            embedding: list[float] = []
            if hasattr(hit, "vectors") and hit.vectors:
                embedding = list(hit.vectors.get("embedding", []))

            return Memory(
                id=doc_id,
                content=fields.get("content", ""),
                embedding=embedding,
                source=fields.get("source", "unknown"),
                memory_type=fields.get("memory_type", "note"),
                tags=json.loads(fields.get("tags_json", "[]")),
                created_at=fields.get("created_at", ""),
                updated_at=fields.get("updated_at", ""),
                metadata=json.loads(fields.get("metadata_json", "{}")),
            )
        except Exception as e:
            logger.warning("Failed to parse Zvec hit: %s", e)
            return None

    def _doc_to_memory(self, doc: Any) -> Memory | None:
        """Convert a Zvec fetched document to a Memory object."""
        try:
            doc_id = doc.id if hasattr(doc, "id") else str(doc.get("id", ""))
            fields = doc.fields if hasattr(doc, "fields") else {}

            embedding: list[float] = []
            if hasattr(doc, "vectors") and doc.vectors:
                embedding = list(doc.vectors.get("embedding", []))

            return Memory(
                id=doc_id,
                content=fields.get("content", ""),
                embedding=embedding,
                source=fields.get("source", "unknown"),
                memory_type=fields.get("memory_type", "note"),
                tags=json.loads(fields.get("tags_json", "[]")),
                created_at=fields.get("created_at", ""),
                updated_at=fields.get("updated_at", ""),
                metadata=json.loads(fields.get("metadata_json", "{}")),
            )
        except Exception as e:
            logger.warning("Failed to parse Zvec doc: %s", e)
            return None
