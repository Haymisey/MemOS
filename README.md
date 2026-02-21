<p align="center">
  <h1 align="center">🧠 MemOS</h1>
  <p align="center"><strong>The Universal Local Context Daemon</strong></p>
  <p align="center">
    A shared memory and knowledge layer for all your AI tools.<br/>
    <em>SQLite for AI memory — zero cloud, zero config, infinite context.</em>
  </p>
</p>

<p align="center">
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-mcp-integration">MCP</a> •
  <a href="#-cli-reference">CLI</a>
</p>

---

## 🎯 The Problem

Your AI tools are suffering from **collective amnesia**.

- **Cursor** doesn't know what you told **Claude**.
- **OpenClaw** can't see what code you wrote in **VS Code**.
- Every tool reinvents its own memory — flat files, proprietary formats, walled gardens.

## 💡 The Solution

**MemOS** is a lightweight daemon that runs silently on your machine, providing a **universal memory layer** that any AI tool can plug into.

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Cursor     │  │ Claude Code │  │   OpenClaw   │
└──────┬───────┘  └──────┬──────┘  └──────┬───────┘
       │                 │                 │
       └────────┐        │        ┌────────┘
                │        │        │
            ┌───▼────────▼────────▼───┐
            │        🧠 MemOS          │
            │   Universal Memory API   │
            │                          │
            │  ┌─────────┐ ┌────────┐  │
            │  │  Zvec    │ │  KG    │  │
            │  │ Vectors  │ │ Graph  │  │
            │  └─────────┘ └────────┘  │
            └──────────────────────────┘
```

## ✨ Key Features

- **🔌 Pluggable Backend** — Zvec (default), LanceDB, ChromaDB via clean ABC
- **🧲 Semantic Search** — Embed anything, search by meaning not keywords
- **🔗 Knowledge Graph** — Track entities and relationships across your work
- **📋 Clipboard Watcher** — Auto-capture copied text as memories
- **📁 File Watcher** — Auto-ingest file changes from watched directories
- **🌐 REST API** — Any tool can store/query via HTTP
- **🤖 MCP Server** — Native integration with Claude, Cursor, Gemini
- **💻 Beautiful CLI** — Git-like commands with rich terminal output
- **🔒 100% Local** — No cloud, no API keys, your data never leaves your machine

## 🚀 Quickstart

### Installation

```bash
pip install -e .
```

### Store Your First Memory

```bash
# Via CLI
memos add "MemOS uses Zvec as its default vector database" --source cli --tags ai,infrastructure

# Search semantically
memos search "what vector database does memos use"
```

### Start the Daemon

```bash
# Start in background
memos start

# Or run in foreground
memos start --foreground

# Check status
memos status

# Stop
memos stop
```

### Use the REST API

```bash
# Store a memory
curl -X POST http://localhost:11437/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "Python 3.12 added type parameter syntax", "source": "api", "tags": ["python"]}'

# Semantic search
curl -X POST http://localhost:11437/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python type hints", "top_k": 5}'

# Health check
curl http://localhost:11437/v1/health
```

## 🏗️ Architecture

```
memos/
├── core/                    # 🧠 Engine Layer
│   ├── base.py              # VectorStoreBackend ABC + data models
│   ├── zvec_backend.py      # Zvec implementation (default)
│   ├── embeddings.py        # EmbeddingEngine (all-MiniLM-L6-v2, 384d)
│   ├── memory_engine.py     # MemoryEngine orchestrator
│   ├── knowledge_graph.py   # Entity-relationship graph
│   └── config.py            # MemOSConfig
│
├── api/                     # 🌐 REST API
│   ├── server.py            # FastAPI application
│   └── models.py            # Pydantic request/response models
│
├── mcp_server/              # 🤖 Model Context Protocol
│   └── server.py            # MCP tools + resources
│
├── cli/                     # 💻 CLI (Typer + Rich)
│   └── main.py              # Git-like commands
│
└── connectors/              # 🔌 Data Connectors
    ├── file_watcher.py      # Auto-ingest file changes
    └── clipboard_watcher.py # Auto-capture clipboard
```

### Pluggable Backend Design

```python
class VectorStoreBackend(ABC):
    """Any backend implements this interface."""
    def initialize(self) -> None: ...
    def add(self, id, content, embedding, ...) -> str: ...
    def search(self, embedding, top_k, filters) -> list[SearchResult]: ...
    def get(self, id) -> Memory | None: ...
    def update(self, id, ...) -> bool: ...
    def delete(self, id) -> bool: ...
    def list_all(self, filters) -> list[Memory]: ...
    def count(self) -> int: ...
    def close(self) -> None: ...
```

Swap backends with zero code changes:
```python
# Default: Zvec
engine = MemoryEngine(config)

# Future: LanceDB
config.backend = "lancedb"
engine = MemoryEngine(config)
```

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/memories` | Store a new memory |
| `POST` | `/v1/memories/search` | Semantic search |
| `GET` | `/v1/memories/{id}` | Get memory by ID |
| `PUT` | `/v1/memories/{id}` | Update a memory |
| `DELETE` | `/v1/memories/{id}` | Delete a memory |
| `POST` | `/v1/entities` | Add KG entity |
| `POST` | `/v1/relationships` | Add KG relationship |
| `POST` | `/v1/graph/search` | Search KG |
| `GET` | `/v1/health` | Health + stats |

Interactive docs: `http://localhost:11437/docs`

## 🤖 MCP Integration

Add MemOS to your MCP client config:

```json
{
  "mcpServers": {
    "memos": {
      "command": "python",
      "args": ["-m", "memos.mcp_server.server"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `memos_store` | Store a memory |
| `memos_search` | Semantic search |
| `memos_add_entity` | Add KG entity |
| `memos_get_related` | Traverse KG |

### MCP Resources

| URI | Description |
|-----|-------------|
| `memos://status` | Daemon status + stats |
| `memos://recent` | Recently stored memories |

## 💻 CLI Reference

```
🧠 MemOS — The Universal Local Context Daemon

Commands:
  start     🚀 Start the MemOS daemon
  stop      🛑 Stop the daemon
  status    📊 Show status and statistics
  add       💾 Store a new memory
  search    🔍 Semantic search
  list      📋 List stored memories
  entity    🔗 Knowledge graph operations
  version   ℹ️  Show version
```

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Vector DB | **Zvec** (Alibaba) | Embedded, zero-infra, blazing fast |
| Embeddings | `all-MiniLM-L6-v2` (384d) | Local, ~80MB, no API keys |
| API | **FastAPI** | Async, auto-docs |
| CLI | **Typer** + **Rich** | Beautiful, git-like UX |
| AI Protocol | **MCP** | Open standard for AI tools |
| File Watch | **watchdog** | OS-native file events |
| Clipboard | **pyperclip** | Cross-platform clipboard access |

## 📄 License

MIT — build the future, freely.

---

<!-- <p align="center">
  <strong>Whoever builds the standard open-source "Memory Layer" that all agents plug into<br/>will own the infrastructure of the AI agent decade.</strong>
</p> -->
