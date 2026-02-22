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
    client = httpx.Client(base_url=url, timeout=30.0)
    try:
        # Quick health check
        resp = client.get("/v1/health")
        if resp.status_code == 200:
            return client
    except (httpx.ConnectError, httpx.HTTPError):
        pass
    client.close()
    return None


# ---------------------------------------------------------------------------
# Daemon Commands
# ---------------------------------------------------------------------------


@app.command()
def start(
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="API host"),
    port: int = typer.Option(11437, "--port", "-p", help="API port"),
) -> None:
    """🚀 Start the MemOS daemon (REST API + MCP server)."""
    config = _get_config()
    config.api_host = host
    config.api_port = port
    config.ensure_dirs()

    if not foreground:
        # Check if already running
        if config.pid_file.exists():
            try:
                pid = int(config.pid_file.read_text().strip())
                # Check if process is still alive
                os.kill(pid, 0)
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
        rprint(Panel(
            f"[green]Starting MemOS daemon...[/green]\n\n"
            f"  🌐 API:  [cyan]http://{host}:{port}[/cyan]\n"
            f"  📁 Data: [dim]{config.data_dir}[/dim]\n"
            f"  🔧 Backend: [magenta]{config.backend}[/magenta]",
            title="🧠 MemOS",
            border_style="green",
        ))

        # Launch daemon process
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        popen_kwargs = {
            "stdout": open(str(config.log_file), "a", encoding="utf-8"),
            "stderr": subprocess.STDOUT,
            "env": env,
            "start_new_session": True,
        }

        if os.name == "nt":
            # On Windows, use these flags to ensure the process survives when the terminal closes
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | 
                subprocess.DETACHED_PROCESS
            )

        proc = subprocess.Popen(
            [sys.executable, "-m", "memos.cli.main", "start", "--foreground",
             "--host", host, "--port", str(port)],
            **popen_kwargs
        )

        config.pid_file.write_text(str(proc.pid))
        rprint(f"\n  [dim]PID: {proc.pid} | Logs: {config.log_file}[/dim]")
        rprint(f"  [green]✓[/green] MemOS is [bold green]running[/bold green]!")
        return

    # Foreground mode — run uvicorn directly
    rprint(Panel(
        f"[green]MemOS daemon starting in foreground...[/green]\n"
        f"  Press [bold]Ctrl+C[/bold] to stop.",
        title="🧠 MemOS",
        border_style="green",
    ))

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
    """📊 Show daemon status and statistics."""
    config = _get_config()

    # Check if daemon is running
    running = False
    pid = None
    if config.pid_file.exists():
        try:
            pid = int(config.pid_file.read_text().strip())
            if os.name == "nt":
                # Windows doesn't support os.kill(pid, 0) reliably for non-child processes
                # and can throw SystemError in some cases.
                try:
                    import ctypes
                    handle = ctypes.windll.kernel32.OpenProcess(1, False, pid)
                    if handle:
                        ctypes.windll.kernel32.CloseHandle(handle)
                        running = True
                except Exception:
                    # Fallback to checking if the PID is actually alive via tasklist if ctypes fails
                    res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], 
                                       capture_output=True, text=True)
                    running = str(pid) in res.stdout
            else:
                os.kill(pid, 0)
                running = True
        except (ProcessLookupError, ValueError, OSError):
            pass

    # Try to get stats from engine
    try:
        engine = _get_engine()
        stats = engine.stats()
        engine.close()
    except Exception:
        stats = None

    # Build status table
    table = Table(title="🧠 MemOS Status", border_style="cyan", show_header=False)
    table.add_column("Key", style="bold cyan", width=20)
    table.add_column("Value", style="white")

    status_text = "[bold green]● Running[/bold green]" if running else "[bold red]● Stopped[/bold red]"
    table.add_row("Daemon", status_text)

    if pid:
        table.add_row("PID", str(pid))

    table.add_row("Version", __version__)
    table.add_row("Data Dir", str(config.data_dir))
    table.add_row("Backend", config.backend)

    if stats:
        table.add_row("Memories", f"[bold]{stats['memory_count']}[/bold]")
        table.add_row("Embedding Model", stats["embedding_model"])
        table.add_row("Embedding Dim", str(stats["embedding_dim"]))

        uptime = stats["uptime_seconds"]
        if uptime > 3600:
            table.add_row("Uptime", f"{uptime / 3600:.1f} hours")
        elif uptime > 60:
            table.add_row("Uptime", f"{uptime / 60:.1f} minutes")
        else:
            table.add_row("Uptime", f"{uptime:.0f} seconds")

    console.print()
    console.print(table)
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
    table.add_column("Tags", style="yellow", width=15)

    for m in memories[:limit]:
        content_preview = m.content[:47] + "..." if len(m.content) > 50 else m.content
        tags = ", ".join(m.tags[:3]) if m.tags else "-"
        table.add_row(m.id, content_preview, m.source, m.memory_type, tags)

    console.print()
    console.print(table)
    if len(memories) > limit:
        console.print(f"\n  [dim]...and {len(memories) - limit} more. Use --limit to show more.[/dim]")
    console.print()


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


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
