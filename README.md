# 🧠 MemOS

### **The Universal Local Context Daemon for the AI Agent Era**

AI agents currently suffer from **collective amnesia**. Your IDE (Cursor) doesn't know what you told your Chat interface (Claude), and your command-line tools have no idea what you're working on in your browser.

**MemOS** is the missing link. It's a lightweight, blazing-fast background daemon that provides a **shared, semantic memory layer** for every AI tool on your machine.

---

## ⚡ The Elevator Pitch
**Whoever owns the local context layer owns the agent decade.** MemOS is building the open-source standard for local AI memory. 
- **100% Local**: No cloud, no API keys, total privacy.
- **Autonomous**: Ingests your files, clipboard, and terminal errors silently.
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
# Start MemOS with clipboard monitoring
memos start --clipboard
```

### 3. Initialize Shell Integration (New! ✨)
Automatically capture terminal errors without changing your workflow.
```bash
# For PowerShell
memos init powershell >> $PROFILE

# For Bash
memos init bash >> ~/.bashrc

# For Zsh
memos init zsh >> ~/.zshrc
```

---

## 🕵️‍♂️ Autonomous Ingestors (Connectors)

### 🐚 Terminal Error Catcher (v1.2)
The ultimate safety net. MemOS monitors your command exit codes.
- **Invisible**: No need to prefix commands. It works natively in your shell.
- **Auto-Capture**: If a command fails, the command string is instantly saved to memory.
- **Contextual**: When you ask your AI "What just happened?", it has the exact failure context.

### 📋 Clipboard Watcher
Auto-captures copied text as memories.
- **Security Check**: Hardened heuristic filtering blocks API keys, tokens, and secrets.
- **Deduplication**: SHA-256 hashing ensures no redundant memories.
- **Filter**: Only captures meaningful text (30+ characters).

### 📁 File Watcher
Indexes your codebases in real-time.
- **Smart**: Respects `.gitignore` and ignores `node_modules`, `.git`, `.venv`.
- **Debounced**: Only saves when you've finished typing.

---

## 💎 Temporal Decay (v1.1)

MemOS doesn't just store everything forever; it manages its own focus.
- **Auto-Expiry**: Transient context (like clipboard entries) expires after 72 hours.
- **Pinning**: Important memories can be "pinned" to stay permanent: `memos pin <id>`.
- **Garbage Collection**: A background task periodically purges outdated context to prevent "vector pollution."

---

## 🤖 AI Agent Integration (MCP)

MemOS implements the **Model Context Protocol (MCP)**, allowing it to plug directly into your favorite AI tools as a semantic knowledge base.

### 🧩 Cursor / VS Code Integration
1. Open Cursor Settings -> **Features** -> **MCP**.
2. Click **+ Add New MCP Server**.
3. **Name**: `memos`, **Type**: `command`, **Command**: `memos mcp`.

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

## 🏗️ Architecture
- **Backend**: **LanceDB** (Embedded, zero-infra vector storage).
- **Embeddings**: Local `all-MiniLM-L6-v2` (384-dim, ~80MB).
- **Security**: Robust regex-based secret filtering for developer safety.

---

## 💻 CLI Reference

| Command | Action |
|---------|--------|
| `memos start` | Launch the context daemon |
| `memos stop` | Kill the daemon |
| `memos status` | View active connectors & stats |
| `memos init` | ✨ Setup shell error catching |
| `memos pin` | 💎 Make a memory permanent |
| `memos unpin` | 🔓 Allow a memory to expire |
| `memos delete` | 🗑️ Manually remove a memory |
| `memos add` | Manually store a memory |
| `memos search` | Instant semantic search |
| `memos watch` | Manage directory intake |
| `memos mcp` | stdio entry point for IDEs |

---

## 📄 License
MIT. Build the future of local-first AI.
