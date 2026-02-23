# 🧠 MemOS

### **The Universal Local Context Daemon for the AI Agent Era**

AI agents currently suffer from **collective amnesia**. Your IDE (Cursor) doesn't know what you told your Chat interface (Claude), and your command-line tools have no idea what you're working on in your browser.

**MemOS** is the missing link. It's a lightweight, blazing-fast background daemon that provides a **shared, semantic memory layer** for every AI tool on your machine.

---

## ⚡ The Elevator Pitch
**Whoever owns the local context layer owns the agent decade.** MemOS is building the open-source standard for local AI memory. 
- **100% Local**: No cloud, no API keys, total privacy.
- **Autonomous**: Ingests your files and clipboard silently while you work.
- **Universal**: One API (REST + MCP) to rule them all.

---

## 🚀 Quickstart

### 1. Install
```bash
# Clone and install locally
pip install -e .
```

### 2. Start the Daemon
```bash
# Start MemOS in the background
memos start --clipboard
```
*MemOS is now monitoring your clipboard and ready to ingest file changes.*

### 3. Check Status
```bash
memos status
```

---

## 🤖 AI Agent Integration (MCP)

MemOS implements the **Model Context Protocol (MCP)**, allowing it to plug directly into your favorite AI tools as a semantic knowledge base.

### 🧩 Cursor / VS Code Integration
1. Open Cursor Settings.
2. Go to **Features** -> **MCP**.
3. Click **+ Add New MCP Server**.
4. **Name**: `memos`
5. **Type**: `command`
6. **Command**: 
   ```bash
   memos mcp
   ```

### 🧡 Claude Desktop Integration
Add the following to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "memos": {
      "command": "memos",
      "args": ["mcp"]
    }
  }
}
```

---

## 🕵️‍♂️ Autonomous Ingestors (Connectors)

MemOS works silently in the background so you don't have to manually "add" context.

### 📋 Clipboard Watcher
Auto-captures copied text as memories.
- **Security Check**: Automatically ignores API keys, passwords, and UUIDs.
- **Deduplication**: SHA-256 hashing ensures you never store the same memory twice.
- **Filter**: Only captures meaningful text (30+ characters).

### 📁 File Watcher
Indexes your codebases in real-time.
```bash
# Watch a directory
memos watch add .
```
- **Smart**: Respects `.gitignore` and ignores `node_modules`, `.git`, `.venv`.
- **Debounced**: Only saves when you've finished typing.

---

## 🏗️ Architecture

MemOS is built for speed and stability on Windows:
- **Backend**: **LanceDB** (Embedded, zero-infra vector storage).
- **Embeddings**: Local `all-MiniLM-L6-v2` (384-dim, ~80MB).
- **Daemon**: Detached background process with a persistent REST API.
- **Bridge**: Remote-First MCP bridge to ensure data consistency.

---

## 💻 CLI Reference

| Command | Action |
|---------|--------|
| `memos start` | Launch the context daemon |
| `memos stop` | Kill the daemon |
| `memos status` | View active connectors & stats |
| `memos add` | Manually store a memory |
| `memos search` | Instant semantic search |
| `memos watch` | Manage directory intake |
| `memos mcp` | stdio entry point for IDEs |

---

## 📄 License
MIT. Build the future of local-first AI.
