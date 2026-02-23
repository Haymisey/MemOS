"""MemOS MCP Server — Model Context Protocol integration.

Exposes MemOS as MCP tools and resources so that any MCP‑compatible AI client
(Claude, Cursor, Gemini, etc.) can read/write memories and traverse the
knowledge graph.

Run with:
    python -m memos.mcp_server.server
Or via the MCP CLI:
    mcp run memos.mcp_server.server
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    TextContent,
    Tool,
)
import httpx
from memos import __version__
from memos.core.base import SearchFilter
from memos.core.config import MemOSConfig
from memos.core.knowledge_graph import KnowledgeGraph
from memos.core.memory_engine import MemoryEngine

logger = logging.getLogger("memos.mcp_server")

# ---------------------------------------------------------------------------
# Global State & Helpers
# ---------------------------------------------------------------------------

server = Server("memos")
_engine: MemoryEngine | None = None
_kg: KnowledgeGraph | None = None
_config: MemOSConfig = MemOSConfig()

def _get_api_client() -> httpx.AsyncClient | None:
    """Check if the daemon is running and return an async client if so."""
    url = f"http://{_config.api_host}:{_config.api_port}"
    try:
        # Simple health check (blocking check for existence, but we use sync for sanity)
        import httpx as sync_httpx
        with sync_httpx.Client(timeout=0.5) as client:
            resp = client.get(f"{url}/v1/health")
            if resp.status_code == 200:
                return httpx.AsyncClient(base_url=url, timeout=30.0)
    except Exception:
        pass
    return None

def _ensure_engine() -> MemoryEngine:
    global _engine
    if _engine is None:
        _engine = MemoryEngine(config=_config)
        _engine.initialize()
    return _engine

def _ensure_kg() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph(config=_config)
        _kg.initialize()
    return _kg


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the tools available to MCP clients."""
    return [
        Tool(
            name="memos_store",
            description="Store a new memory in MemOS. Content is automatically embedded for semantic search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content to store as a memory",
                    },
                    "source": {
                        "type": "string",
                        "description": "Where this memory came from (e.g., 'cursor', 'claude', 'user')",
                        "default": "mcp",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Category of memory (note, code, conversation, file)",
                        "default": "note",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for organization",
                        "default": [],
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="memos_search",
            description="Search MemOS for memories semantically related to a query. Returns the most relevant stored memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional: filter results by source",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Optional: filter results by type",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memos_add_entity",
            description="Add an entity to the MemOS knowledge graph (e.g., a person, project, concept).",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Entity name",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Entity type (person, project, concept, file, etc.)",
                        "default": "concept",
                    },
                    "properties": {
                        "type": "object",
                        "description": "Optional key-value properties for the entity",
                        "default": {},
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="memos_get_related",
            description="Find all knowledge graph relationships involving a given entity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "The entity name to search for relationships",
                    },
                    "relationship_type": {
                        "type": "string",
                        "description": "Optional: filter by relationship type",
                    },
                },
                "required": ["entity_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle MCP tool calls. Prioritizes REST API if daemon is running."""
    client = _get_api_client()
    
    try:
        if name == "memos_store":
            if client:
                resp = await client.post("/v1/memories", json={
                    "content": arguments["content"],
                    "source": arguments.get("source", "mcp"),
                    "memory_type": arguments.get("memory_type", "note"),
                    "tags": arguments.get("tags", []),
                })
                data = resp.json()
                return [TextContent(type="text", text=json.dumps(data))]
            else:
                engine = _ensure_engine()
                memory = engine.store(
                    content=arguments["content"],
                    source=arguments.get("source", "mcp"),
                    memory_type=arguments.get("memory_type", "note"),
                    tags=arguments.get("tags", []),
                )
                return [TextContent(type="text", text=json.dumps({
                    "status": "stored",
                    "id": memory.id,
                    "content_length": len(memory.content),
                    "source": memory.source,
                    "memory_type": memory.memory_type,
                }))]

        elif name == "memos_search":
            if client:
                resp = await client.post("/v1/memories/search", json={
                    "query": arguments["query"],
                    "top_k": arguments.get("top_k", 5),
                    "filters": {
                        "source": arguments.get("source"),
                        "memory_type": arguments.get("memory_type"),
                    }
                })
                data = resp.json()
                return [TextContent(type="text", text=json.dumps(data))]
            else:
                engine = _ensure_engine()
                filters = SearchFilter(
                    source=arguments.get("source"),
                    memory_type=arguments.get("memory_type"),
                )
                results = engine.search(
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 5),
                    filters=filters,
                )
                return [TextContent(type="text", text=json.dumps({
                    "query": arguments["query"],
                    "count": len(results),
                    "results": [
                        {
                            "id": r.memory.id, "content": r.memory.content,
                            "source": r.memory.source, "memory_type": r.memory.memory_type,
                            "tags": r.memory.tags, "score": r.score,
                        } for r in results
                    ],
                }))]

        elif name == "memos_add_entity":
            if client:
                resp = await client.post("/v1/entities", json={
                    "name": arguments["name"],
                    "entity_type": arguments.get("entity_type", "concept"),
                    "properties": arguments.get("properties", {}),
                })
                data = resp.json()
                return [TextContent(type="text", text=json.dumps(data))]
            else:
                kg = _ensure_kg()
                entity = kg.add_entity(
                    name=arguments["name"],
                    entity_type=arguments.get("entity_type", "concept"),
                    properties=arguments.get("properties", {}),
                )
                return [TextContent(type="text", text=json.dumps({
                    "status": "created", "id": entity.id,
                    "name": entity.name, "entity_type": entity.entity_type,
                }))]

        elif name == "memos_get_related":
            if client:
                resp = await client.post("/v1/graph/search", json={
                    "entity_name": arguments["entity_name"],
                    "relationship_type": arguments.get("relationship_type"),
                })
                data = resp.json()
                return [TextContent(type="text", text=json.dumps(data))]
            else:
                kg = _ensure_kg()
                relationships = kg.get_related(
                    entity_name=arguments["entity_name"],
                    relationship_type=arguments.get("relationship_type"),
                )
                return [TextContent(type="text", text=json.dumps({
                    "entity": arguments["entity_name"],
                    "count": len(relationships),
                    "relationships": [
                        {
                            "id": r.id, "from": r.from_entity,
                            "to": r.to_entity, "type": r.relationship_type,
                        } for r in relationships
                    ],
                }))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    finally:
        if client:
            await client.aclose()


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------


@server.list_resources()
async def list_resources() -> list[Resource]:
    """Declare available MCP resources."""
    return [
        Resource(
            uri="memos://status",
            name="MemOS Status",
            description="Current daemon status, memory count, and configuration",
            mimeType="application/json",
        ),
        Resource(
            uri="memos://recent",
            name="Recent Memories",
            description="The most recently stored memories",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read an MCP resource. Prioritizes REST API if daemon is running."""
    client = _get_api_client()
    try:
        if uri == "memos://status":
            if client:
                resp = await client.get("/v1/health")
                return json.dumps(resp.json())
            else:
                engine = _ensure_engine()
                stats = engine.stats()
                return json.dumps({"version": __version__, **stats})

        elif uri == "memos://recent":
            engine = _ensure_engine()
            memories = engine.list_memories()
            recent = memories[-10:] if len(memories) > 10 else memories
            return json.dumps({
                "count": len(recent),
                "memories": [
                    {
                        "id": m.id, "content": m.content[:200],
                        "source": m.source, "memory_type": m.memory_type,
                        "created_at": m.created_at,
                    } for m in reversed(recent)
                ],
            })
        
        return json.dumps({"error": f"Unknown resource: {uri}"})
    finally:
        if client:
            await client.aclose()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


async def main():
    """Run the MCP server over stdio."""
    logger.info("Starting MemOS MCP Server v%s", __version__)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
