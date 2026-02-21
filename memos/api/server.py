"""MemOS REST API — FastAPI server.

Exposes the MemoryEngine and KnowledgeGraph over HTTP for any tool to consume.

Start with:
    uvicorn memos.api.server:create_app --host 127.0.0.1 --port 11437
Or via the CLI:
    memos start
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from memos import __version__
from memos.api.models import (
    AddEntityRequest,
    AddRelationshipRequest,
    EntityResponse,
    GraphSearchRequest,
    HealthResponse,
    MemoryResponse,
    RelationshipResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    StoreMemoryRequest,
    SuccessResponse,
    UpdateMemoryRequest,
)
from memos.core.base import SearchFilter
from memos.core.config import MemOSConfig
from memos.core.knowledge_graph import KnowledgeGraph
from memos.core.memory_engine import MemoryEngine

logger = logging.getLogger("memos.api.server")

# ---------------------------------------------------------------------------
# Global engine instances (initialized during app lifespan)
# ---------------------------------------------------------------------------

_engine: MemoryEngine | None = None
_knowledge_graph: KnowledgeGraph | None = None


def _get_engine() -> MemoryEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="MemOS engine not initialized")
    return _engine


def _get_kg() -> KnowledgeGraph:
    if _knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    return _knowledge_graph


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------


def create_app(config: MemOSConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional MemOS configuration. Defaults to standard config.

    Returns:
        Configured FastAPI instance.
    """
    cfg = config or MemOSConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize engines on startup, clean up on shutdown."""
        global _engine, _knowledge_graph

        logger.info("Starting MemOS API server...")
        _engine = MemoryEngine(config=cfg)
        _engine.initialize()

        _knowledge_graph = KnowledgeGraph(config=cfg)
        _knowledge_graph.initialize()

        logger.info("MemOS API server ready on %s:%d", cfg.api_host, cfg.api_port)
        yield

        # Shutdown
        logger.info("Shutting down MemOS API server...")
        if _engine:
            _engine.close()
        if _knowledge_graph:
            _knowledge_graph.close()

    app = FastAPI(
        title="MemOS",
        description="The Universal Local Context Daemon — shared memory for all your AI tools.",
        version=__version__,
        lifespan=lifespan,
    )

    # Allow local tools to connect
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_memory_routes(app)
    _register_graph_routes(app)
    _register_system_routes(app)

    return app


# ---------------------------------------------------------------------------
# Memory Routes
# ---------------------------------------------------------------------------


def _register_memory_routes(app: FastAPI) -> None:
    @app.post("/v1/memories", response_model=MemoryResponse, tags=["Memories"])
    async def store_memory(req: StoreMemoryRequest):
        """Store a new memory. Content is automatically embedded."""
        engine = _get_engine()
        memory = engine.store(
            content=req.content,
            source=req.source,
            memory_type=req.memory_type,
            tags=req.tags,
            metadata=req.metadata,
        )
        return MemoryResponse(
            id=memory.id,
            content=memory.content,
            source=memory.source,
            memory_type=memory.memory_type,
            tags=memory.tags,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            metadata=memory.metadata,
        )

    @app.post("/v1/memories/search", response_model=SearchResponse, tags=["Memories"])
    async def search_memories(req: SearchRequest):
        """Semantic search for memories related to a query."""
        engine = _get_engine()
        filters = SearchFilter(
            source=req.source,
            memory_type=req.memory_type,
            tags=req.tags,
        )
        results = engine.search(query=req.query, top_k=req.top_k, filters=filters)

        return SearchResponse(
            results=[
                SearchResultResponse(
                    memory=MemoryResponse(
                        id=r.memory.id,
                        content=r.memory.content,
                        source=r.memory.source,
                        memory_type=r.memory.memory_type,
                        tags=r.memory.tags,
                        created_at=r.memory.created_at,
                        updated_at=r.memory.updated_at,
                        metadata=r.memory.metadata,
                    ),
                    score=r.score,
                )
                for r in results
            ],
            count=len(results),
            query=req.query,
        )

    @app.get("/v1/memories/{memory_id}", response_model=MemoryResponse, tags=["Memories"])
    async def get_memory(memory_id: str):
        """Retrieve a specific memory by ID."""
        engine = _get_engine()
        memory = engine.get(memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
        return MemoryResponse(
            id=memory.id,
            content=memory.content,
            source=memory.source,
            memory_type=memory.memory_type,
            tags=memory.tags,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            metadata=memory.metadata,
        )

    @app.put("/v1/memories/{memory_id}", response_model=SuccessResponse, tags=["Memories"])
    async def update_memory(memory_id: str, req: UpdateMemoryRequest):
        """Update an existing memory. Re-embeds if content changes."""
        engine = _get_engine()
        success = engine.update(
            memory_id=memory_id,
            content=req.content,
            source=req.source,
            memory_type=req.memory_type,
            tags=req.tags,
            metadata=req.metadata,
        )
        if not success:
            raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
        return SuccessResponse(success=True, message=f"Memory '{memory_id}' updated")

    @app.delete("/v1/memories/{memory_id}", response_model=SuccessResponse, tags=["Memories"])
    async def delete_memory(memory_id: str):
        """Delete a memory by ID."""
        engine = _get_engine()
        success = engine.delete(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
        return SuccessResponse(success=True, message=f"Memory '{memory_id}' deleted")


# ---------------------------------------------------------------------------
# Knowledge Graph Routes
# ---------------------------------------------------------------------------


def _register_graph_routes(app: FastAPI) -> None:
    @app.post("/v1/entities", response_model=EntityResponse, tags=["Knowledge Graph"])
    async def add_entity(req: AddEntityRequest):
        """Add an entity to the knowledge graph."""
        kg = _get_kg()
        entity = kg.add_entity(
            name=req.name,
            entity_type=req.entity_type,
            properties=req.properties,
        )
        return EntityResponse(
            id=entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            properties=entity.properties,
            created_at=entity.created_at,
        )

    @app.post("/v1/relationships", response_model=RelationshipResponse, tags=["Knowledge Graph"])
    async def add_relationship(req: AddRelationshipRequest):
        """Add a relationship between two entities."""
        kg = _get_kg()
        rel = kg.add_relationship(
            from_entity=req.from_entity,
            to_entity=req.to_entity,
            relationship_type=req.relationship_type,
            properties=req.properties,
        )
        return RelationshipResponse(
            id=rel.id,
            from_entity=rel.from_entity,
            to_entity=rel.to_entity,
            relationship_type=rel.relationship_type,
            properties=rel.properties,
            created_at=rel.created_at,
        )

    @app.post(
        "/v1/graph/search",
        response_model=list[RelationshipResponse],
        tags=["Knowledge Graph"],
    )
    async def search_graph(req: GraphSearchRequest):
        """Search the knowledge graph for relationships involving an entity."""
        kg = _get_kg()
        relationships = kg.get_related(
            entity_name=req.entity_name,
            relationship_type=req.relationship_type,
            top_k=req.top_k,
        )
        return [
            RelationshipResponse(
                id=r.id,
                from_entity=r.from_entity,
                to_entity=r.to_entity,
                relationship_type=r.relationship_type,
                properties=r.properties,
                created_at=r.created_at,
            )
            for r in relationships
        ]


# ---------------------------------------------------------------------------
# System Routes
# ---------------------------------------------------------------------------


def _register_system_routes(app: FastAPI) -> None:
    @app.get("/v1/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """Health check with engine statistics."""
        engine = _get_engine()
        stats = engine.stats()
        return HealthResponse(
            status="ok",
            version=__version__,
            memory_count=stats["memory_count"],
            uptime_seconds=stats["uptime_seconds"],
            backend=stats["backend"],
            embedding_model=stats["embedding_model"],
            embedding_dim=stats["embedding_dim"],
            data_dir=stats["data_dir"],
        )
