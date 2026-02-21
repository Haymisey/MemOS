"""Knowledge Graph layer for MemOS.

Manages entities (people, projects, files, concepts) and the relationships
between them, stored in separate Zvec collections. This enables graph-like
traversal on top of the flat memory store.

Example:
    kg = KnowledgeGraph(config)
    kg.initialize()
    kg.add_entity("Python", entity_type="language", properties={"version": "3.12"})
    kg.add_entity("MemOS", entity_type="project")
    kg.add_relationship("MemOS", "Python", relationship_type="uses")
    related = kg.get_related("MemOS")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import zvec

from memos.core.base import generate_id
from memos.core.config import MemOSConfig
from memos.core.embeddings import EmbeddingEngine

logger = logging.getLogger("memos.core.knowledge_graph")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A node in the knowledge graph.

    Attributes:
        id:          Unique entity ID.
        name:        Human-readable name.
        entity_type: Category (e.g., "person", "project", "concept", "file").
        properties:  Arbitrary key-value properties.
        created_at:  ISO timestamp of creation.
    """

    id: str
    name: str
    entity_type: str = "concept"
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Relationship:
    """An edge in the knowledge graph.

    Attributes:
        id:                 Unique relationship ID.
        from_entity:        Source entity name.
        to_entity:          Target entity name.
        relationship_type:  Type of relationship (e.g., "uses", "knows", "created").
        properties:         Arbitrary key-value properties.
        created_at:         ISO timestamp of creation.
    """

    id: str
    from_entity: str
    to_entity: str
    relationship_type: str = "related_to"
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """Entity-relationship knowledge graph backed by Zvec (or LanceDB on Windows).

    Uses two separate collections/tables:
    - `entities`:      Stores entity nodes with vector embeddings of their names.
    - `relationships`: Stores relationship edges with metadata.

    Args:
        config: MemOS configuration for paths and settings.
    """

    def __init__(self, config: MemOSConfig | None = None) -> None:
        self.config = config or MemOSConfig()
        self._embedder = EmbeddingEngine(model_name=self.config.embedding_model)
        self._entities_col: Any = None
        self._relationships_col: Any = None
        self._db: Any = None
        self._initialized = False

    def initialize(self) -> None:
        """Create or open the entity and relationship storage."""
        if self._initialized:
            return

        self.config.ensure_dirs()
        
        if self.config.backend == "zvec":
            self._init_zvec()
        else:
            self._init_lancedb()
            
        self._initialized = True
        logger.info("KnowledgeGraph initialized (backend=%s).", self.config.backend)

    def _init_zvec(self) -> None:
        """Set up Zvec collections."""
        import zvec
        
        # Entities
        path = str(self.config.entities_path)
        schema = zvec.CollectionSchema(
            name="entities",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, self.config.embedding_dim),
            fields=[
                zvec.ScalarSchema("name", zvec.DataType.STRING),
                zvec.ScalarSchema("entity_type", zvec.DataType.STRING),
                zvec.ScalarSchema("properties_json", zvec.DataType.STRING),
                zvec.ScalarSchema("created_at", zvec.DataType.STRING),
            ],
        )
        try:
            self._entities_col = zvec.open(path=path)
        except Exception:
            self.config.entities_path.mkdir(parents=True, exist_ok=True)
            self._entities_col = zvec.create_and_open(path=path, schema=schema)

        # Relationships
        path = str(self.config.relationships_path)
        schema = zvec.CollectionSchema(
            name="relationships",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, self.config.embedding_dim),
            fields=[
                zvec.ScalarSchema("from_entity", zvec.DataType.STRING),
                zvec.ScalarSchema("to_entity", zvec.DataType.STRING),
                zvec.ScalarSchema("relationship_type", zvec.DataType.STRING),
                zvec.ScalarSchema("properties_json", zvec.DataType.STRING),
                zvec.ScalarSchema("created_at", zvec.DataType.STRING),
            ],
        )
        try:
            self._relationships_col = zvec.open(path=path)
        except Exception:
            self.config.relationships_path.mkdir(parents=True, exist_ok=True)
            self._relationships_col = zvec.create_and_open(path=path, schema=schema)

    def _init_lancedb(self) -> None:
        """Set up LanceDB tables."""
        import lancedb
        import pyarrow as pa
        
        db_path = self.config.db_path / "kg"
        db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(db_path))
        
        # Entities Schema
        e_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), self.config.embedding_dim)),
            pa.field("name", pa.string()),
            pa.field("entity_type", pa.string()),
            pa.field("properties_json", pa.string()),
            pa.field("created_at", pa.string()),
        ])
        
        if "entities" in self._db.table_names():
            self._entities_col = self._db.open_table("entities")
        else:
            self._entities_col = self._db.create_table("entities", schema=e_schema)
            
        # Relationships Schema
        r_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), self.config.embedding_dim)),
            pa.field("from_entity", pa.string()),
            pa.field("to_entity", pa.string()),
            pa.field("relationship_type", pa.string()),
            pa.field("properties_json", pa.string()),
            pa.field("created_at", pa.string()),
        ])
        
        if "relationships" in self._db.table_names():
            self._relationships_col = self._db.open_table("relationships")
        else:
            self._relationships_col = self._db.create_table("relationships", schema=r_schema)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("KnowledgeGraph not initialized. Call initialize() first.")

    # -------------------------------------------------------------------
    # Entity Operations
    # -------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str = "concept",
        properties: dict[str, Any] | None = None,
    ) -> Entity:
        self._ensure_initialized()
        entity_id = generate_id()
        now = datetime.now(timezone.utc).isoformat()
        embedding = self._embedder.embed_text(f"{entity_type}: {name}")
        props_json = json.dumps(properties or {})

        if self.config.backend == "zvec":
            import zvec
            doc = zvec.Doc(
                id=entity_id,
                vectors={"embedding": embedding},
                fields={"name": name, "entity_type": entity_type, "properties_json": props_json, "created_at": now},
            )
            self._entities_col.insert(doc)
        else:
            self._entities_col.add([{
                "id": entity_id, "embedding": embedding, "name": name, 
                "entity_type": entity_type, "properties_json": props_json, "created_at": now
            }])

        logger.info("Added entity '%s' (type=%s, id=%s)", name, entity_type, entity_id)
        return Entity(id=entity_id, name=name, entity_type=entity_type, properties=properties or {}, created_at=now)

    def search_entities(
        self,
        query: str,
        top_k: int = 5,
        entity_type: str | None = None,
    ) -> list[Entity]:
        self._ensure_initialized()
        embedding = self._embedder.embed_text(query)
        entities: list[Entity] = []

        if self.config.backend == "zvec":
            import zvec
            query_kwargs: dict[str, Any] = {
                "vectors": zvec.VectorQuery(field_name="embedding", vector=embedding),
                "topk": top_k,
            }
            if entity_type:
                query_kwargs["filter"] = f'entity_type = "{entity_type}"'
            results = self._entities_col.query(**query_kwargs)
            for hit in results:
                f = hit.fields if hasattr(hit, "fields") else {}
                entities.append(Entity(
                    id=hit.id if hasattr(hit, "id") else "",
                    name=f.get("name", ""),
                    entity_type=f.get("entity_type", "concept"),
                    properties=json.loads(f.get("properties_json", "{}")),
                    created_at=f.get("created_at", ""),
                ))
        else:
            q = self._entities_col.search(embedding).limit(top_k)
            if entity_type:
                q = q.where(f'entity_type = "{entity_type}"')
            results = q.to_arrow().to_pydict()
            for i in range(len(results["id"])):
                entities.append(Entity(
                    id=results["id"][i],
                    name=results["name"][i],
                    entity_type=results["entity_type"][i],
                    properties=json.loads(results["properties_json"][i]),
                    created_at=results["created_at"][i],
                ))

        return entities

    # -------------------------------------------------------------------
    # Relationship Operations
    # -------------------------------------------------------------------

    def add_relationship(
        self,
        from_entity: str,
        to_entity: str,
        relationship_type: str = "related_to",
        properties: dict[str, Any] | None = None,
    ) -> Relationship:
        self._ensure_initialized()
        rel_id = generate_id()
        now = datetime.now(timezone.utc).isoformat()
        description = f"{from_entity} {relationship_type} {to_entity}"
        embedding = self._embedder.embed_text(description)
        props_json = json.dumps(properties or {})

        if self.config.backend == "zvec":
            import zvec
            doc = zvec.Doc(
                id=rel_id,
                vectors={"embedding": embedding},
                fields={
                    "from_entity": from_entity, "to_entity": to_entity,
                    "relationship_type": relationship_type, "properties_json": props_json, "created_at": now
                },
            )
            self._relationships_col.insert(doc)
        else:
            self._relationships_col.add([{
                "id": rel_id, "embedding": embedding, "from_entity": from_entity, "to_entity": to_entity,
                "relationship_type": relationship_type, "properties_json": props_json, "created_at": now
            }])

        logger.info("Added relationship: %s -[%s]-> %s", from_entity, relationship_type, to_entity)
        return Relationship(
            id=rel_id, from_entity=from_entity, to_entity=to_entity,
            relationship_type=relationship_type, properties=properties or {}, created_at=now
        )

    def get_related(
        self,
        entity_name: str,
        relationship_type: str | None = None,
        top_k: int = 10,
    ) -> list[Relationship]:
        self._ensure_initialized()
        relationships: list[Relationship] = []

        if self.config.backend == "zvec":
            f_parts = [f'(from_entity = "{entity_name}" OR to_entity = "{entity_name}")']
            if relationship_type: f_parts.append(f'relationship_type = "{relationship_type}"')
            try:
                results = self._relationships_col.query(filter=" AND ".join(f_parts), topk=top_k)
            except Exception:
                embedding = self._embedder.embed_text(entity_name)
                results = self._relationships_col.query(
                    vectors={"embedding": embedding}, topk=top_k
                )
            for hit in results:
                f = hit.fields if hasattr(hit, "fields") else {}
                relationships.append(Relationship(
                    id=hit.id if hasattr(hit, "id") else "",
                    from_entity=f.get("from_entity", ""),
                    to_entity=f.get("to_entity", ""),
                    relationship_type=f.get("relationship_type", "related_to"),
                    properties=json.loads(f.get("properties_json", "{}")),
                    created_at=f.get("created_at", ""),
                ))
        else:
            # LanceDB SQL filter
            where = f'(from_entity = "{entity_name}" OR to_entity = "{entity_name}")'
            if relationship_type:
                where += f' AND relationship_type = "{relationship_type}"'
            
            results = self._relationships_col.search().where(where).limit(top_k).to_arrow().to_pydict()
            for i in range(len(results["id"])):
                relationships.append(Relationship(
                    id=results["id"][i],
                    from_entity=results["from_entity"][i],
                    to_entity=results["to_entity"][i],
                    relationship_type=results["relationship_type"][i],
                    properties=json.loads(results["properties_json"][i]),
                    created_at=results["created_at"][i],
                ))

        return relationships

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Close collections."""
        if self.config.backend == "zvec":
            for col in (self._entities_col, self._relationships_col):
                if col is not None: col.close()
        self._entities_col = None
        self._relationships_col = None
        self._initialized = False
        logger.info("KnowledgeGraph shut down.")
