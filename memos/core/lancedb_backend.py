"""LanceDB backend implementation for the VectorStoreBackend ABC.

This serves as a robust fallback for Windows users, as Zvec is currently
only available for Linux and macOS. LanceDB is also an embedded, serverless
vector database ("SQLite for embeddings").
"""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from memos.core.base import Memory, SearchFilter, SearchResult, VectorStoreBackend, UNDEFINED

logger = logging.getLogger("memos.core.lancedb_backend")


class LanceDBBackend(VectorStoreBackend):
    """LanceDB implementation of the VectorStoreBackend interface.

    Storage layout:
        {db_path}/lancedb/   — main directory
        Memories are stored in a table named 'memories'.
    """

    def __init__(self, db_path: Path, embedding_dim: int = 384) -> None:
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._db = None
        self._table = None

    def initialize(self) -> None:
        """Initialize LanceDB connection and ensure table exists."""
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._db_path))

        # Define Arrow schema for the table
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), self._embedding_dim)),
            pa.field("source", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
            pa.field("expires_at", pa.string()),  # ISO string or null
            pa.field("is_pinned", pa.bool_()),    # Boolean
            pa.field("metadata", pa.string()),    # JSON string
        ])

        if "memories" in self._db.table_names():
            self._table = self._db.open_table("memories")
            
            # Check if we need to migrate schema (v1.0 -> v1.1)
            existing_names = self._table.schema.names
            if "expires_at" not in existing_names or "is_pinned" not in existing_names:
                logger.info("Migrating LanceDB schema to v1.1...")
                # Load all data as Arrow table
                table_data = self._table.to_arrow()
                
                # Add columns if missing using PyArrow
                if "expires_at" not in existing_names:
                    # Create a column of null strings
                    null_array = pa.array([None] * len(table_data), type=pa.string())
                    table_data = table_data.append_column("expires_at", null_array)
                
                if "is_pinned" not in existing_names:
                    # Create a column of False values
                    false_array = pa.array([False] * len(table_data), type=pa.bool_())
                    table_data = table_data.append_column("is_pinned", false_array)
                
                # Re-create table with new schema and data
                self._db.drop_table("memories")
                self._table = self._db.create_table("memories", data=table_data, schema=schema)
                logger.info("Successfully migrated LanceDB schema using PyArrow.")
        else:
            self._table = self._db.create_table("memories", schema=schema)
        
        logger.info("LanceDB backend initialized at %s", self._db_path)

    def _ensure_table(self):
        if self._table is None:
            raise RuntimeError("LanceDBBackend not initialized. Call initialize() first.")
        return self._table

    def add(
        self,
        id: str,
        content: str,
        embedding: list[float],
        source: str = "unknown",
        memory_type: str = "note",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
        is_pinned: bool = False,
    ) -> str:
        table = self._ensure_table()
        
        now = datetime.now(timezone.utc).isoformat()
        
        data = {
            "id": id,
            "content": content,
            "embedding": embedding,
            "source": source,
            "memory_type": memory_type,
            "tags": tags or [],
            "created_at": now,
            "updated_at": now,
            "expires_at": expires_at,
            "is_pinned": is_pinned,
            "metadata": json.dumps(metadata or {}),
        }
        
        table.add([data])
        return id

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: SearchFilter | None = None,
    ) -> list[SearchResult]:
        table = self._ensure_table()
        
        query = table.search(embedding).limit(top_k)
        
        # Apply filters if provided
        where_clause = self._build_where_clause(filters)
        if where_clause:
            query = query.where(where_clause)
        
        results = query.to_arrow()
        
        output = []
        for i in range(len(results)):
            row = results.slice(i, 1).to_pydict()
            
            memory = Memory(
                id=row["id"][0],
                content=row["content"][0],
                embedding=row["embedding"][0],
                source=row["source"][0],
                memory_type=row["memory_type"][0],
                tags=row["tags"][0],
                created_at=row["created_at"][0],
                updated_at=row["updated_at"][0],
                expires_at=row["expires_at"][0] if "expires_at" in row else None,
                is_pinned=row["is_pinned"][0] if "is_pinned" in row and row["is_pinned"][0] is not None else False,
                metadata=json.loads(row["metadata"][0]),
            )
            # LanceDB results include _distance which is L2 distance
            output.append(SearchResult(memory=memory, score=1.0 - (row.get("_distance", [0])[0] / 2.0)))
            
        return output

    def get(self, id: str) -> Memory | None:
        table = self._ensure_table()
        
        res = table.search().where(f'id = "{id}"').limit(1).to_arrow()
        if len(res) == 0:
            return None
            
        row = res.to_pydict()
        return Memory(
            id=row["id"][0],
            content=row["content"][0],
            embedding=row["embedding"][0],
            source=row["source"][0],
            memory_type=row["memory_type"][0],
            tags=row["tags"][0],
            created_at=row["created_at"][0],
            updated_at=row["updated_at"][0],
            expires_at=row["expires_at"][0] if "expires_at" in row else None,
            is_pinned=row["is_pinned"][0] if "is_pinned" in row and row["is_pinned"][0] is not None else False,
            metadata=json.loads(row["metadata"][0]),
        )

    def update(
        self,
        id: str,
        content: str | None = None,
        embedding: list[float] | None = None,
        source: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = UNDEFINED,
        is_pinned: bool | None = UNDEFINED,
    ) -> bool:
        existing = self.get(id)
        if not existing:
            return False
            
        table = self._ensure_table()
        now = datetime.now(timezone.utc).isoformat()
        
        new_data = {
            "id": id,
            "content": content if content is not None else existing.content,
            "embedding": embedding if embedding is not None else existing.embedding,
            "source": source if source is not None else existing.source,
            "memory_type": memory_type if memory_type is not None else existing.memory_type,
            "tags": tags if tags is not None else existing.tags,
            "created_at": existing.created_at,
            "updated_at": now,
            "expires_at": expires_at if expires_at is not UNDEFINED else existing.expires_at,
            "is_pinned": is_pinned if is_pinned is not UNDEFINED else existing.is_pinned,
            "metadata": json.dumps(metadata if metadata is not None else existing.metadata),
        }
        
        table.delete(f'id = "{id}"')
        table.add([new_data])
        return True

    def delete(self, id: str) -> bool:
        table = self._ensure_table()
        table.delete(f'id = "{id}"')
        return True

    def list_all(self, filters: SearchFilter | None = None) -> list[Memory]:
        table = self._ensure_table()
        
        query = table.search()
        where_clause = self._build_where_clause(filters)
        if where_clause:
            query = query.where(where_clause)
            
        results = query.to_arrow()
        
        output = []
        for i in range(len(results)):
            row = results.slice(i, 1).to_pydict()
            memory = Memory(
                id=row["id"][0],
                content=row["content"][0],
                embedding=row["embedding"][0],
                source=row["source"][0],
                memory_type=row["memory_type"][0],
                tags=row["tags"][0],
                created_at=row["created_at"][0],
                updated_at=row["updated_at"][0],
                expires_at=row["expires_at"][0] if "expires_at" in row else None,
                is_pinned=row["is_pinned"][0] if "is_pinned" in row and row["is_pinned"][0] is not None else False,
                metadata=json.loads(row["metadata"][0]),
            )
            output.append(memory)
        return output

    def count(self) -> int:
        table = self._ensure_table()
        return len(table)

    def cleanup_expired(self) -> int:
        """Purge expired, unpinned memories."""
        table = self._ensure_table()
        now = datetime.now(timezone.utc).isoformat()
        
        # We delete where expires_at < now AND is_pinned == False
        where = f'expires_at IS NOT NULL AND expires_at < "{now}" AND is_pinned = false'
        
        # Determine count before delete
        expired_count = len(table.search().where(where).to_arrow())
        if expired_count > 0:
            table.delete(where)
            logger.info("Purged %d expired memories", expired_count)
            
        return expired_count

    def close(self) -> None:
        pass

    def _build_where_clause(self, filters: SearchFilter | None) -> str | None:
        if not filters:
            return None
            
        clauses = []
        if filters.source:
            clauses.append(f'source = "{filters.source}"')
        if filters.memory_type:
            clauses.append(f'memory_type = "{filters.memory_type}"')
            
        if not clauses:
            return None
        return " AND ".join(clauses)
