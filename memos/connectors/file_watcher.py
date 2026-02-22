import logging
import threading
import time
from pathlib import Path

import pathspec
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

# Default ignore patterns (common heavy directories)
DEFAULT_IGNORES = [
    "node_modules/",
    ".venv/",
    "venv/",
    ".git/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    "dist/",
    "build/",
    "*.pyc",
    "*.log",
]


class _MemOSFileHandler(FileSystemEventHandler):
    """Handle file modification events with strict ignore rules and debouncing."""

    def __init__(self, engine, watch_dir: Path, debounce: float = DEBOUNCE_SECONDS) -> None:
        super().__init__()
        self._engine = engine
        self._watch_dir = watch_dir
        self._debounce = debounce
        self._last_seen: dict[str, float] = {}
        self._spec = self._load_ignore_spec()

    def _load_ignore_spec(self) -> pathspec.PathSpec:
        """Load patterns from .gitignore and add default ignores."""
        patterns = list(DEFAULT_IGNORES)
        gitignore = self._watch_dir / ".gitignore"
        if gitignore.exists():
            try:
                patterns.extend(gitignore.read_text(encoding="utf-8").splitlines())
            except Exception as e:
                logger.warning("Failed to read .gitignore in %s: %s", self._watch_dir, e)
        
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return

        path = Path(event.src_path)
        
        # 1. Check extension
        if path.suffix.lower() not in WATCHED_EXTENSIONS:
            return

        # 2. Check ignores
        try:
            relative_path = path.relative_to(self._watch_dir)
            if self._spec.match_file(str(relative_path)):
                return
        except Exception:
            # Fallback if relative_path fails for some reason
            if any(part in DEFAULT_IGNORES or part.startswith(".") for part in path.parts):
                return

        # 3. Debounce
        now = time.time()
        last = self._last_seen.get(str(path), 0)
        if now - last < self._debounce:
            return
        self._last_seen[str(path)] = now

        # 4. Read and ingest (deferred slightly to ensure file is unlocked)
        threading.Timer(0.5, self._ingest, args=[path]).start()

    def _ingest(self, path: Path) -> None:
        """Performed after a short delay to ensure write completion."""
        try:
            if not path.exists():
                return
            
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
            logger.info("Ingested file: %s (%d chars)", path.name, len(content))
        except Exception as e:
            logger.debug("Failed to ingest %s (expected if still locked): %s", path.name, e)


class FileWatcher:
    """Watch directories for changes with .gitignore support and heavy-dir filtering."""

    def __init__(self, engine, watch_dirs: list[Path] | None = None) -> None:
        self._engine = engine
        self._watch_dirs = watch_dirs or []
        self._observer: Observer | None = None
        self._running = False

    def start(self) -> None:
        """Start watching directories."""
        if self._running or not self._watch_dirs:
            return

        self._observer = Observer()
        for watch_dir in self._watch_dirs:
            watch_dir = Path(watch_dir).expanduser().resolve()
            if watch_dir.exists() and watch_dir.is_dir():
                handler = _MemOSFileHandler(self._engine, watch_dir)
                self._observer.schedule(handler, str(watch_dir), recursive=True)
                logger.info("Watching: %s", watch_dir)

        self._observer.start()
        self._running = True
        logger.info("FileWatcher started (%d dirs)", len(self._watch_dirs))

    def stop(self) -> None:
        """Stop watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self._running = False
        logger.info("FileWatcher stopped.")

    @property
    def is_running(self) -> bool:
        return self._running
