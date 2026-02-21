"""File Watcher connector for MemOS.

Uses watchdog to monitor directories and auto-ingest file changes as memories.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("memos.connectors.file_watcher")

# File extensions to watch for content changes
WATCHED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".json",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".bash",
    ".html", ".css", ".sql", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".rb", ".php", ".swift", ".kt", ".scala", ".r",
}

# Maximum file size to ingest (1 MB)
MAX_FILE_SIZE = 1_048_576

# Debounce: ignore rapid repeated changes to the same file (seconds)
DEBOUNCE_SECONDS = 5.0


class _MemOSFileHandler(FileSystemEventHandler):
    """Handle file modification events and queue them for ingestion."""

    def __init__(self, engine, debounce: float = DEBOUNCE_SECONDS) -> None:
        super().__init__()
        self._engine = engine
        self._debounce = debounce
        self._last_seen: dict[str, float] = {}

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Only watch recognized file types
        if path.suffix.lower() not in WATCHED_EXTENSIONS:
            return

        # Skip hidden files and directories
        if any(part.startswith(".") for part in path.parts):
            return

        # Debounce: skip if we saw this file very recently
        now = time.time()
        last = self._last_seen.get(str(path), 0)
        if now - last < self._debounce:
            return
        self._last_seen[str(path)] = now

        # Read and ingest
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                return

            content = path.read_text(encoding="utf-8", errors="ignore")
            if len(content.strip()) < 20:
                return

            self._engine.store(
                content=content,
                source="file_watcher",
                memory_type="file",
                tags=[path.suffix.lstrip(".")],
                metadata={"file_path": str(path), "file_name": path.name},
            )
            logger.info("Ingested file change: %s (%d chars)", path.name, len(content))
        except Exception as e:
            logger.warning("Failed to ingest %s: %s", path, e)


class FileWatcher:
    """Watch directories for file changes and auto-ingest into MemOS.

    Example:
        watcher = FileWatcher(engine, watch_dirs=[Path("~/projects")])
        watcher.start()
        # ... runs in background ...
        watcher.stop()
    """

    def __init__(self, engine, watch_dirs: list[Path] | None = None) -> None:
        self._engine = engine
        self._watch_dirs = watch_dirs or []
        self._observer: Observer | None = None
        self._running = False

    def start(self) -> None:
        """Start watching directories in a background thread."""
        if self._running:
            return

        if not self._watch_dirs:
            logger.info("No directories configured for file watching.")
            return

        self._observer = Observer()
        handler = _MemOSFileHandler(self._engine)

        for watch_dir in self._watch_dirs:
            if watch_dir.exists() and watch_dir.is_dir():
                self._observer.schedule(handler, str(watch_dir), recursive=True)
                logger.info("Watching directory: %s", watch_dir)

        self._observer.start()
        self._running = True
        logger.info("FileWatcher started (%d directories).", len(self._watch_dirs))

    def stop(self) -> None:
        """Stop watching directories."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self._running = False
        logger.info("FileWatcher stopped.")

    @property
    def is_running(self) -> bool:
        return self._running
