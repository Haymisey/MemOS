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
    """Entity-relationship knowledge graph backed by Zvec.

    Uses two separate Zvec collections:
    - `entities`:      Stores entity nodes with vector embeddings of their names.
    - `relationships`: Stores relationship edges with metadata.

    Args:
        config: MemOS configuration for paths and settings.
    """

    def __init__(self, config: MemOSConfig | None = None) -> None:
        self.config = config or MemOSConfig()
        self._embedder = EmbeddingEngine(model_name=self.config.embedding_model)
        self._entities_collection: zvec.Collection | None = None
        self._relationships_collection: zvec.Collection | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Create or open the entity and relationship collections."""
        if self._initialized:
            return

        self.config.ensure_dirs()
        self._init_entities_collection()
        self._init_relationships_collection()
        self._initialized = True
        logger.info("KnowledgeGraph initialized.")

    def _init_entities_collection(self) -> None:
        """Set up the entities Zvec collection."""
        path = str(self.config.entities_path)

        schema = zvec.CollectionSchema(
            name="entities",
            vectors=zvec.VectorSchema(
                "embedding", zvec.DataType.VECTOR_FP32, self.config.embedding_dim
            ),
            fields=[
                zvec.ScalarSchema("name", zvec.DataType.STRING),
                zvec.ScalarSchema("entity_type", zvec.DataType.STRING),
                zvec.ScalarSchema("properties_json", zvec.DataType.STRING),
                zvec.ScalarSchema("created_at", zvec.DataType.STRING),
            ],
        )

        try:
            self._entities_collection = zvec.open(path=path)
        except Exception:
            self.config.entities_path.mkdir(parents=True, exist_ok=True)
            self._entities_collection = zvec.create_and_open(path=path, schema=schema)

    def _init_relationships_collection(self) -> None:
        """Set up the relationships Zvec collection."""
        path = str(self.config.relationships_path)

        schema = zvec.CollectionSchema(
            name="relationships",
            vectors=zvec.VectorSchema(
                "embedding", zvec.DataType.VECTOR_FP32, self.config.embedding_dim
            ),
            fields=[
                zvec.ScalarSchema("from_entity", zvec.DataType.STRING),
                zvec.ScalarSchema("to_entity", zvec.DataType.STRING),
                zvec.ScalarSchema("relationship_type", zvec.DataType.STRING),
                zvec.ScalarSchema("properties_json", zvec.DataType.STRING),
                zvec.ScalarSchema("created_at", zvec.DataType.STRING),
            ],
        )

        try:
            self._relationships_collection = zvec.open(path=path)
        except Exception:
            self.config.relationships_path.mkdir(parents=True, exist_ok=True)
            self._relationships_collection = zvec.create_and_open(path=path, schema=schema)

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
        """Add a new entity to the knowledge graph.

        Args:
            name:        Human-readable entity name.
            entity_type: Category (e.g., "person", "project", "concept").
            properties:  Optional key-value properties.

        Returns:
            The created Entity object.
        """
        self._ensure_initialized()
        assert self._entities_collection is not None

        entity_id = generate_id()
        now = datetime.now(timezone.utc).isoformat()

        # Embed the entity name for semantic search
        embedding = self._embedder.embed_text(f"{entity_type}: {name}")

        doc = zvec.Doc(
            id=entity_id,
            vectors={"embedding": embedding},
            fields={
                "name": name,
                "entity_type": entity_type,
                "properties_json": json.dumps(properties or {}),
                "created_at": now,
            },
        )

        self._entities_collection.insert(doc)
        logger.info("Added entity '%s' (type=%s, id=%s)", name, entity_type, entity_id)

        return Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            created_at=now,
        )

    def search_entities(
        self,
        query: str,
        top_k: int = 5,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Search for entities by semantic similarity.

        Args:
            query:       Natural language search query.
            top_k:       Maximum number of results.
            entity_type: Optional filter by entity type.

        Returns:
            List of matching Entity objects.
        """
        self._ensure_initialized()
        assert self._entities_collection is not None

        embedding = self._embedder.embed_text(query)

        query_kwargs: dict[str, Any] = {
            "vectors": zvec.VectorQuery(field_name="embedding", vector=embedding),
            "topk": top_k,
        }
        if entity_type:
            query_kwargs["filter"] = f'entity_type = "{entity_type}"'

        results = self._entities_collection.query(**query_kwargs)

        entities: list[Entity] = []
        for hit in results:
            fields = hit.fields if hasattr(hit, "fields") else {}
            entities.append(
                Entity(
                    id=hit.id if hasattr(hit, "id") else "",
                    name=fields.get("name", ""),
                    entity_type=fields.get("entity_type", "concept"),
                    properties=json.loads(fields.get("properties_json", "{}")),
                    created_at=fields.get("created_at", ""),
                )
            )

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
        """Add a relationship (edge) between two entities.

        Args:
            from_entity:       Source entity name.
            to_entity:         Target entity name.
            relationship_type: Type of relationship (e.g., "uses", "knows").
            properties:        Optional key-value properties.

        Returns:
            The created Relationship object.
        """
        self._ensure_initialized()
        assert self._relationships_collection is not None

        rel_id = generate_id()
        now = datetime.now(timezone.utc).isoformat()

        # Embed the relationship description for semantic search
        description = f"{from_entity} {relationship_type} {to_entity}"
        embedding = self._embedder.embed_text(description)

        doc = zvec.Doc(
            id=rel_id,
            vectors={"embedding": embedding},
            fields={
                "from_entity": from_entity,
                "to_entity": to_entity,
                "relationship_type": relationship_type,
                "properties_json": json.dumps(properties or {}),
                "created_at": now,
            },
        )

        self._relationships_collection.insert(doc)
        logger.info(
            "Added relationship: %s -[%s]-> %s (id=%s)",
            from_entity, relationship_type, to_entity, rel_id,
        )

        return Relationship(
            id=rel_id,
            from_entity=from_entity,
            to_entity=to_entity,
            relationship_type=relationship_type,
            properties=properties or {},
            created_at=now,
        )

    def get_related(
        self,
        entity_name: str,
        relationship_type: str | None = None,
        top_k: int = 10,
    ) -> list[Relationship]:
        """Find all relationships involving a given entity.

        Args:
            entity_name:       The entity to search for.
            relationship_type: Optional filter by relationship type.
            top_k:             Maximum number of results.

        Returns:
            List of Relationship objects involving the entity.
        """
        self._ensure_initialized()
        assert self._relationships_collection is not None

        # Build filter for entity name (either side of the relationship)
        filter_parts: list[str] = []
        filter_parts.append(
            f'(from_entity = "{entity_name}" OR to_entity = "{entity_name}")'
        )
        if relationship_type:
            filter_parts.append(f'relationship_type = "{relationship_type}"')

        filter_str = " AND ".join(filter_parts)

        try:
            results = self._relationships_collection.query(
                filter=filter_str, topk=top_k
            )
        except Exception:
            # Fallback: if OR filters not supported, do semantic search
            embedding = self._embedder.embed_text(entity_name)
            results = self._relationships_collection.query(
                vectors=zvec.VectorQuery(field_name="embedding", vector=embedding),
                topk=top_k,
            )

        relationships: list[Relationship] = []
        for hit in results:
            fields = hit.fields if hasattr(hit, "fields") else {}

            # Filter results to only include this entity
            from_e = fields.get("from_entity", "")
            to_e = fields.get("to_entity", "")
            if entity_name not in (from_e, to_e):
                continue

            relationships.append(
                Relationship(
                    id=hit.id if hasattr(hit, "id") else "",
                    from_entity=from_e,
                    to_entity=to_e,
                    relationship_type=fields.get("relationship_type", "related_to"),
                    properties=json.loads(fields.get("properties_json", "{}")),
                    created_at=fields.get("created_at", ""),
                )
            )

        return relationships

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Close all collections and release resources."""
        for col in (self._entities_collection, self._relationships_collection):
            if col is not None:
                try:
                    col.close()
                except Exception:
                    pass
        self._entities_collection = None
        self._relationships_collection = None
        self._initialized = False
        logger.info("KnowledgeGraph shut down.")
