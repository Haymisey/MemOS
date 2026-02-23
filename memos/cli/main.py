"""MemOS CLI — Git-like command line interface with Typer + Rich.

Usage:
    memos start              Start the daemon (API + MCP server)
    memos stop               Stop the daemon
    memos status             Show daemon status
    memos add "content"      Store a new memory
    memos search "query"     Semantic search for memories
    memos entity add "name"  Add a knowledge graph entity
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from memos import __version__
from memos.core.base import SearchFilter
from memos.core.config import MemOSConfig
from memos.core.memory_engine import MemoryEngine

console = Console()
app = typer.Typer(
    name="memos",
    help="🧠 MemOS — The Universal Local Context Daemon",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Sub-command group for entities
entity_app = typer.Typer(help="🔗 Knowledge graph entity operations", no_args_is_help=True)
app.add_typer(entity_app, name="entity")

# Sub-command group for watching
watch_app = typer.Typer(help="👁️  Manage directory watching", no_args_is_help=True)
app.add_typer(watch_app, name="watch")


def _get_config() -> MemOSConfig:
    """Load MemOS configuration."""
    return MemOSConfig()


def _get_engine() -> MemoryEngine:
    """Create and initialize a MemoryEngine instance."""
    config = _get_config()
    engine = MemoryEngine(config=config)
    engine.initialize()
    return engine


def _get_api_client() -> httpx.Client | None:
    """Return a healthy API client if the daemon is running, else None."""
    config = _get_config()
    url = f"http://{config.api_host}:{config.api_port}"
    client = httpx.Client(base_url=url, timeout=5.0) # Shorter timeout for health check
    try:
        resp = client.get("/v1/health")
        if resp.status_code == 200:
            return client
    except Exception:
        pass
    client.close()
    return None


@watch_app.command(name="add")
def watch_add(path: Path = typer.Argument(..., help="Directory to watch")):
    """👁️  Add a directory to the auto-ingestion watcher."""
    config = _get_config()
    path = path.expanduser().resolve()
    if not path.is_dir():
        rprint(f"[red]Error:[/red] {path} is not a directory.")
        raise typer.Exit(1)
    
    if path not in config.watch_dirs:
        config.watch_dirs.append(path)
        config.save()
        rprint(f"[green]Added to watch list:[/green] {path}")
        rprint("[dim]Restart the daemon for changes to take effect.[/dim]")
    else:
        rprint(f"[yellow]Already watching:[/yellow] {path}")


@watch_app.command(name="list")
def watch_list():
    """📋 List all directories currently being watched."""
    config = _get_config()
    if not config.watch_dirs:
        rprint("[yellow]No directories are being watched.[/yellow]")
        return
    
    table = Table(title="👁️  Watched Directories", border_style="cyan")
    table.add_column("Path", style="white")
    for d in config.watch_dirs:
        table.add_row(str(d))
    console.print(table)


# ---------------------------------------------------------------------------
# Daemon Commands
# ---------------------------------------------------------------------------


@app.command()
def start(
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
    host: str = typer.Option(None, "--host", "-h", help="API host (overrides config)"),
    port: int = typer.Option(None, "--port", "-p", help="API port (overrides config)"),
    clipboard: bool = typer.Option(None, "--clipboard/--no-clipboard", help="Enable/disable clipboard watching"),
) -> None:
    """🚀 Start the MemOS daemon (REST API + MCP server)."""
    config = _get_config()
    
    if host: config.api_host = host
    if port: config.api_port = port
    if clipboard is not None: config.enable_clipboard = clipboard
    
    config.save()
    config.ensure_dirs()
    
    host = config.api_host
    port = config.api_port

    if not foreground:
        # ... (rest of start remains the same, but use popen_kwargs)
        # Check if already running
        if config.pid_file.exists():
            try:
                pid = int(config.pid_file.read_text().strip())
                if os.name == "nt":
                    import ctypes
                    handle = ctypes.windll.kernel32.OpenProcess(1, False, pid)
                    running = bool(handle)
                    if handle: ctypes.windll.kernel32.CloseHandle(handle)
                else:
                    os.kill(pid, 0)
                    running = True
                
                if running:
                    rprint(Panel(
                        f"[yellow]MemOS is already running[/yellow] (PID: {pid})\n"
                        f"Use [bold]memos stop[/bold] to stop it.",
                        title="⚠️  Already Running",
                        border_style="yellow",
                    ))
                    raise typer.Exit(1)
            except (ProcessLookupError, ValueError, OSError):
                config.pid_file.unlink(missing_ok=True)

        # Start in background
        clp_status = "[green]Enabled[/green]" if config.enable_clipboard else "[dim]Disabled[/dim]"
        rprint(Panel(
            f"[green]Starting MemOS daemon...[/green]\n\n"
            f"  🌐 API:  [cyan]http://{host}:{port}[/cyan]\n"
            f"  📁 Data: [dim]{config.data_dir}[/dim]\n"
            f"  🔧 Backend: [magenta]{config.backend}[/magenta]\n"
            f"  📋 Clipboard: {clp_status}\n"
            f"  👁️  Watch Dirs: [bold]{len(config.watch_dirs)}[/bold]",
            title="🧠 MemOS",
            border_style="green",
        ))

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        popen_kwargs = {
            "stdout": open(str(config.log_file), "a", encoding="utf-8"),
            "stderr": subprocess.STDOUT,
            "env": env,
            "start_new_session": True,
        }

        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | 
                subprocess.DETACHED_PROCESS
            )

        proc = subprocess.Popen(
            [sys.executable, "-m", "memos.cli.main", "start", "--foreground"],
            **popen_kwargs
        )

        config.pid_file.write_text(str(proc.pid))
        rprint(f"\n  [dim]PID: {proc.pid} | Logs: {config.log_file}[/dim]")
        rprint(f"  [green]✓[/green] MemOS is [bold green]running[/bold green]!")
        return

    # Foreground mode
    import uvicorn
    from memos.api.server import create_app
    api_app = create_app(config=config)
    uvicorn.run(api_app, host=host, port=port, log_level="info")


@app.command()
def stop() -> None:
    """🛑 Stop the MemOS daemon."""
    config = _get_config()

    if not config.pid_file.exists():
        rprint(Panel(
            "[yellow]MemOS daemon is not running.[/yellow]",
            title="ℹ️  Status",
            border_style="yellow",
        ))
        return

    try:
        pid = int(config.pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        config.pid_file.unlink(missing_ok=True)
        rprint(Panel(
            f"[green]MemOS daemon stopped[/green] (PID: {pid})",
            title="🛑 Stopped",
            border_style="red",
        ))
    except (ProcessLookupError, ValueError, OSError) as e:
        config.pid_file.unlink(missing_ok=True)
        rprint(Panel(
            f"[yellow]Daemon process not found (may have already stopped).[/yellow]\n"
            f"Cleaned up PID file.",
            title="ℹ️  Status",
            border_style="yellow",
        ))


@app.command()
def status() -> None:
    """📊 Show daemon status, statistics, and connector health."""
    config = _get_config()
    client = _get_api_client()
    
    # Check if daemon is running via PID
    running = False
    pid = None
    if config.pid_file.exists():
        try:
            pid = int(config.pid_file.read_text().strip())
            if os.name == "nt":
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(1, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    running = True
            else:
                os.kill(pid, 0)
                running = True
        except (ProcessLookupError, ValueError, OSError):
            pass

    # Fetch stats (prefer API)
    stats = None
    if client:
        try:
            stats = client.get("/v1/health").json()
            running = True # If API is up, it's running
        except Exception:
            pass
        finally:
            client.close()
    
    if stats is None:
        # Fallback to local engine for basic stats
        try:
            engine = _get_engine()
            stats = engine.stats()
            engine.close()
        except Exception:
            pass

    # Build status table
    table = Table(title="🧠 MemOS Status", border_style="cyan", show_header=False)
    table.add_column("Key", style="bold cyan", width=25)
    table.add_column("Value", style="white")

    status_text = "[bold green]● Running[/bold green]" if running else "[bold red]● Stopped[/bold red]"
    table.add_row("Daemon", status_text)

    if pid:
        table.add_row("PID", str(pid))

    table.add_row("Version", __version__)
    table.add_row("Data Dir", str(config.data_dir))
    table.add_row("Backend", stats.get("backend", config.backend) if stats else config.backend)

    if stats:
        table.add_row("Memories", f"[bold]{stats.get('memory_count', 0)}[/bold]")
        table.add_row("Embedding Model", stats.get("embedding_model", "unknown"))
        
        # Connector Status
        conns = stats.get("connectors", {})
        fw = conns.get("file_watcher", {})
        cw = conns.get("clipboard_watcher", {})
        
        fw_status = "[green]Active[/green]" if fw.get("active") else "[dim]Inactive[/dim]"
        cw_status = "[green]Active[/green]" if cw.get("active") else "[dim]Inactive[/dim]"
        
        table.add_row("File Watcher", fw_status)
        if fw.get("dirs"):
            table.add_row("  - Watching", f"{len(fw['dirs'])} directories")
            
        table.add_row("Clipboard Watcher", cw_status)

        uptime = stats.get("uptime_seconds", 0)
        unit = "seconds"
        if uptime > 3600:
            uptime /= 3600
            unit = "hours"
        elif uptime > 60:
            uptime /= 60
            unit = "minutes"
        table.add_row("Uptime", f"{uptime:.1f} {unit}")

    console.print()
    console.print(table)
    
    if stats and stats.get("connectors", {}).get("file_watcher", {}).get("dirs"):
        console.print("[dim]Watched Dirs:[/dim]")
        for d in stats["connectors"]["file_watcher"]["dirs"]:
            console.print(f"  [dim]• {d}[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# Memory Commands
# ---------------------------------------------------------------------------


@app.command()
def add(
    content: str = typer.Argument(..., help="Text content to store as a memory"),
    source: str = typer.Option("cli", "--source", "-s", help="Memory source"),
    memory_type: str = typer.Option("note", "--type", "-t", help="Memory type"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
) -> None:
    """💾 Store a new memory."""
    with console.status("[cyan]Storing memory...[/cyan]", spinner="dots"):
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        # Try API first (instant)
        client = _get_api_client()
        if client:
            try:
                resp = client.post("/v1/memories", json={
                    "content": content,
                    "source": source,
                    "memory_type": memory_type,
                    "tags": tag_list,
                })
                resp.raise_for_status()
                data = resp.json()
                client.close()

                rprint(Panel(
                    f"[green]Memory stored successfully![/green]\n\n"
                    f"  [bold]ID:[/bold]     [cyan]{data['id']}[/cyan]\n"
                    f"  [bold]Source:[/bold] {data['source']}\n"
                    f"  [bold]Type:[/bold]   {data['memory_type']}\n"
                    f"  [bold]Tags:[/bold]   {', '.join(data['tags']) if data['tags'] else '[dim]none[/dim]'}\n"
                    f"  [bold]Size:[/bold]   {len(content)} chars",
                    title="💾 Stored (via API)",
                    border_style="green",
                ))
                return
            except Exception as e:
                console.print(f"[dim]API add failed, falling back to local: {e}[/dim]")

        # Fallback to local engine (requires loading model)
        engine = _get_engine()
        memory = engine.store(
            content=content,
            source=source,
            memory_type=memory_type,
            tags=tag_list,
        )
        engine.close()

    rprint(Panel(
        f"[green]Memory stored successfully![/green]\n\n"
        f"  [bold]ID:[/bold]     [cyan]{memory.id}[/cyan]\n"
        f"  [bold]Source:[/bold] {memory.source}\n"
        f"  [bold]Type:[/bold]   {memory.memory_type}\n"
        f"  [bold]Tags:[/bold]   {', '.join(memory.tags) if memory.tags else '[dim]none[/dim]'}\n"
        f"  [bold]Size:[/bold]   {len(content)} chars",
        title="💾 Stored (Local Fallback)",
        border_style="green",
    ))


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    source: str = typer.Option(None, "--source", "-s", help="Filter by source"),
    memory_type: str = typer.Option(None, "--type", "-t", help="Filter by type"),
) -> None:
    """🔍 Search memories by semantic similarity."""
    with console.status("[cyan]Searching...[/cyan]", spinner="dots"):
        # Try API first (instant)
        client = _get_api_client()
        if client:
            try:
                resp = client.post("/v1/memories/search", json={
                    "query": query,
                    "top_k": top_k,
                    "source": source,
                    "memory_type": memory_type,
                })
                resp.raise_for_status()
                data = resp.json()
                client.close()

                if not data["results"]:
                    rprint(Panel(
                        f"[yellow]No results found for:[/yellow] \"{query}\"",
                        title="🔍 Search (via API)",
                        border_style="yellow",
                    ))
                    return

                rprint(f"\n[bold]🔍 Results for:[/bold] \"{query}\" [dim](via API)[/dim]\n")
                for i, r in enumerate(data["results"], 1):
                    m = r["memory"]
                    score_pct = r["score"] * 100 if r["score"] <= 1 else r["score"]
                    display_content = m["content"][:200] + "..." if len(m["content"]) > 200 else m["content"]
                    
                    score_style = "bold green" if score_pct >= 80 else "bold yellow" if score_pct >= 50 else "bold red"
                    tag_str = ", ".join(m["tags"]) if m["tags"] else "[dim]none[/dim]"
                    
                    rprint(Panel(
                        f"{display_content}\n\n"
                        f"  [dim]ID:[/dim] {m['id']}  "
                        f"[dim]Source:[/dim] {m['source']}  "
                        f"[dim]Type:[/dim] {m['memory_type']}  "
                        f"[dim]Tags:[/dim] {tag_str}",
                        title=f"#{i}  [{score_style}]{score_pct:.1f}% match[/{score_style}]",
                        border_style="cyan",
                    ))
                return
            except Exception as e:
                console.print(f"[dim]API search failed, falling back to local: {e}[/dim]")

        # Fallback to local engine
        engine = _get_engine()
        filters = SearchFilter(source=source, memory_type=memory_type)
        results = engine.search(query=query, top_k=top_k, filters=filters)
        engine.close()

    if not results:
        rprint(Panel(
            f"[yellow]No results found for:[/yellow] \"{query}\"",
            title="🔍 Search (Local Fallback)",
            border_style="yellow",
        ))
        return

    rprint(f"\n[bold]🔍 Results for:[/bold] \"{query}\" [dim](Local Fallback)[/dim]\n")

    for i, result in enumerate(results, 1):
        m = result.memory
        score_pct = result.score * 100 if result.score <= 1 else result.score

        # Truncate content for display
        display_content = m.content[:200] + "..." if len(m.content) > 200 else m.content

        # Color-code by score
        if score_pct >= 80:
            score_style = "bold green"
        elif score_pct >= 50:
            score_style = "bold yellow"
        else:
            score_style = "bold red"

        tag_str = ", ".join(m.tags) if m.tags else "[dim]none[/dim]"

        rprint(Panel(
            f"{display_content}\n\n"
            f"  [dim]ID:[/dim] {m.id}  "
            f"[dim]Source:[/dim] {m.source}  "
            f"[dim]Type:[/dim] {m.memory_type}  "
            f"[dim]Tags:[/dim] {tag_str}",
            title=f"#{i}  [{score_style}]{score_pct:.1f}% match[/{score_style}]",
            border_style="cyan",
        ))


@app.command(name="list")
def list_memories(
    source: str = typer.Option(None, "--source", "-s", help="Filter by source"),
    memory_type: str = typer.Option(None, "--type", "-t", help="Filter by type"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results to show"),
) -> None:
    """📋 List stored memories."""
    with console.status("[cyan]Loading memories...[/cyan]", spinner="dots"):
        # Try API first
        client = _get_api_client()
        if client:
            try:
                # We use the search endpoint with no query but filters if supported,
                # or add a dedicated list endpoint. For now, let's just check the health and use local
                # as listing is usually fast since it doesn't involve embeddings.
                # However, for consistency, let's keep it local for now.
                pass
            except Exception:
                pass

        engine = _get_engine()
        filters = SearchFilter(source=source, memory_type=memory_type)
        memories = engine.list_memories(filters=filters)
        engine.close()

    if not memories:
        rprint(Panel("[yellow]No memories found.[/yellow]", title="📋 List", border_style="yellow"))
        return

    table = Table(title=f"📋 Memories ({len(memories)} total)", border_style="cyan")
    table.add_column("ID", style="cyan", width=14)
    table.add_column("Content", style="white", max_width=50)
    table.add_column("Source", style="magenta", width=12)
    table.add_column("Type", style="green", width=10)
    table.add_column("Pin", style="bold yellow", width=4, justify="center")
    table.add_column("Expiry", style="dim", width=15)
    table.add_column("Tags", style="yellow", width=15)

    for m in memories[:limit]:
        content_preview = m.content[:47] + "..." if len(m.content) > 50 else m.content
        tags = ", ".join(m.tags[:3]) if m.tags else "-"
        is_pinned = "💎" if m.is_pinned else "-"
        
        # Format expiry
        expiry = "-"
        if m.expires_at:
            try:
                from datetime import datetime, timezone
                exp_dt = datetime.fromisoformat(m.expires_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                diff = exp_dt - now
                if diff.total_seconds() < 0:
                    expiry = "[red]Expired[/red]"
                elif diff.days > 0:
                    expiry = f"{diff.days}d {diff.seconds // 3600}h"
                elif diff.seconds > 3600:
                    expiry = f"{diff.seconds // 3600}h {(diff.seconds % 3600) // 60}m"
                else:
                    expiry = f"{diff.seconds // 60}m"
            except Exception:
                expiry = m.expires_at[:10]

        table.add_row(m.id, content_preview, m.source, m.memory_type, is_pinned, expiry, tags)

    console.print()
    console.print(table)
    if len(memories) > limit:
        console.print(f"\n  [dim]...and {len(memories) - limit} more. Use --limit to show more.[/dim]")
    console.print()


@app.command()
def delete(memory_id: str = typer.Argument(..., help="Memory ID to delete")) -> None:
    """🗑️ Delete a memory permanently."""
    with console.status("[red]Deleting memory...[/red]", spinner="dots"):
        # Try API first (instant)
        client = _get_api_client()
        if client:
            try:
                resp = client.delete(f"/v1/memories/{memory_id}")
                if resp.status_code == 200:
                    rprint(f"[green]✔ Memory [bold]{memory_id}[/bold] deleted successfully.[/green]")
                    return
            except Exception:
                pass

        # Fallback to local engine (requires loading model)
        engine = _get_engine()
        if engine.delete(memory_id):
            rprint(f"[green]✔ Memory [bold]{memory_id}[/bold] deleted successfully (local).[/green]")
        else:
            rprint(f"[red]✖ Memory '{memory_id}' not found.[/red]")
        engine.close()

@app.command()
def pin(memory_id: str = typer.Argument(..., help="Memory ID to pin")) -> None:
    """💎 Pin a memory to prevent it from ever expiring."""
    with console.status("[cyan]Pinning memory...[/cyan]", spinner="dots"):
        client = _get_api_client()
        if client:
            try:
                resp = client.post(f"/v1/memories/{memory_id}/pin")
                if resp.status_code == 200:
                    rprint(f"[green]✔ Memory [bold]{memory_id}[/bold] pinned permanently.[/green]")
                    return
            except Exception:
                pass
        
        # Fallback to local
        engine = _get_engine()
        if engine.pin(memory_id):
            rprint(f"[green]✔ Memory [bold]{memory_id}[/bold] pinned permanently (local).[/green]")
        else:
            rprint(f"[red]✖ Memory '{memory_id}' not found.[/red]")
        engine.close()


@app.command()
def unpin(memory_id: str = typer.Argument(..., help="Memory ID to unpin")) -> None:
    """🔓 Unpin a memory, allowing it to expire if it has a TTL."""
    with console.status("[cyan]Unpinning memory...[/cyan]", spinner="dots"):
        client = _get_api_client()
        if client:
            try:
                resp = client.post(f"/v1/memories/{memory_id}/unpin")
                if resp.status_code == 200:
                    rprint(f"[green]✔ Memory [bold]{memory_id}[/bold] unpinned.[/green]")
                    return
            except Exception:
                pass
        
        # Fallback to local
        engine = _get_engine()
        if engine.unpin(memory_id):
            rprint(f"[green]✔ Memory [bold]{memory_id}[/bold] unpinned (local).[/green]")
        else:
            rprint(f"[red]✖ Memory '{memory_id}' not found.[/red]")
        engine.close()


# ---------------------------------------------------------------------------
# Entity Commands
# ---------------------------------------------------------------------------


@entity_app.command(name="add")
def entity_add(
    name: str = typer.Argument(..., help="Entity name"),
    entity_type: str = typer.Option("concept", "--type", "-t", help="Entity type"),
) -> None:
    """➕ Add a knowledge graph entity."""
    from memos.core.knowledge_graph import KnowledgeGraph

    with console.status("[cyan]Adding entity...[/cyan]", spinner="dots"):
        config = _get_config()
        kg = KnowledgeGraph(config=config)
        kg.initialize()
        entity = kg.add_entity(name=name, entity_type=entity_type)
        kg.close()

    rprint(Panel(
        f"[green]Entity added![/green]\n\n"
        f"  [bold]ID:[/bold]   [cyan]{entity.id}[/cyan]\n"
        f"  [bold]Name:[/bold] {entity.name}\n"
        f"  [bold]Type:[/bold] {entity.entity_type}",
        title="🔗 Entity",
        border_style="green",
    ))


@entity_app.command(name="search")
def entity_search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """🔍 Search knowledge graph entities."""
    from memos.core.knowledge_graph import KnowledgeGraph

    with console.status("[cyan]Searching entities...[/cyan]", spinner="dots"):
        config = _get_config()
        kg = KnowledgeGraph(config=config)
        kg.initialize()
        entities = kg.search_entities(query=query, top_k=top_k)
        kg.close()

    if not entities:
        rprint(Panel("[yellow]No entities found.[/yellow]", title="🔍 Search", border_style="yellow"))
        return

    table = Table(title="🔗 Entities", border_style="cyan")
    table.add_column("ID", style="cyan", width=14)
    table.add_column("Name", style="bold white")
    table.add_column("Type", style="magenta")

    for e in entities:
        table.add_row(e.id, e.name, e.entity_type)

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Version & Info
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """ℹ️  Show MemOS version."""
    rprint(Panel(
        f"[bold cyan]MemOS[/bold cyan] v{__version__}\n"
        f"[dim]The Universal Local Context Daemon[/dim]",
        border_style="cyan",
    ))


@app.command()
def mcp() -> None:
    """🔌 Start the MCP server (stdio mode). Use this for Cursor/Claude integration."""
    import asyncio
    from memos.mcp_server.server import main as mcp_main
    
    # Run the async main function
    try:
        asyncio.run(mcp_main())
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
