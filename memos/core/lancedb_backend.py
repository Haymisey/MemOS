"""LanceDB backend implementation for the VectorStoreBackend ABC.

This serves as a robust fallback for Windows users, as Zvec is currently
only available for Linux and macOS. LanceDB is also an embedded, serverless
vector database ("SQLite for embeddings").
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from memos.core.base import Memory, SearchFilter, SearchResult, VectorStoreBackend

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
            pa.field("metadata", pa.string()),  # JSON string
        ])

        if "memories" in self._db.table_names():
            self._table = self._db.open_table("memories")
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
    ) -> str:
        table = self._ensure_table()
        import json
        
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
        import json
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
                metadata=json.loads(row["metadata"][0]),
            )
            # LanceDB results include _distance which is L2 distance
            # We want similarity, but for now we'll just return what we have
            output.append(SearchResult(memory=memory, score=1.0 - (row.get("_distance", [0])[0] / 2.0)))
            
        return output

    def get(self, id: str) -> Memory | None:
        table = self._ensure_table()
        import json
        
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
    ) -> bool:
        existing = self.get(id)
        if not existing:
            return False
            
        table = self._ensure_table()
        import json
        
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
            "metadata": json.dumps(metadata if metadata is not None else existing.metadata),
        }
        
        # LanceDB update is usually delete + add or merge
        table.delete(f'id = "{id}"')
        table.add([new_data])
        return True

    def delete(self, id: str) -> bool:
        table = self._ensure_table()
        table.delete(f'id = "{id}"')
        return True

    def list_all(self, filters: SearchFilter | None = None) -> list[Memory]:
        table = self._ensure_table()
        import json
        
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
                metadata=json.loads(row["metadata"][0]),
            )
            output.append(memory)
        return output

    def count(self) -> int:
        table = self._ensure_table()
        return len(table)

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
            
        # LanceDB SQL-like syntax for array contains is a bit specific
        # For simplicity in v1, we skip tags in the where clause or use post-filtering
        
        if not clauses:
            return None
        return " AND ".join(clauses)
