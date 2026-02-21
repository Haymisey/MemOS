"""Clipboard Watcher connector for MemOS.

Runs in a background thread, polls the clipboard every few seconds, and
auto-ingests new copied text into the memory store.
"""

from __future__ import annotations

import logging
import threading
import time

import pyperclip

logger = logging.getLogger("memos.connectors.clipboard_watcher")

# Minimum length of clipboard text to trigger ingestion
MIN_CLIPBOARD_LENGTH = 20

# How often to poll the clipboard (seconds)
POLL_INTERVAL = 3.0

# Maximum clipboard text to ingest (100 KB — skip huge copy-pastes)
MAX_CLIPBOARD_LENGTH = 102_400


class ClipboardWatcher:
    """Passively watch the clipboard and auto-ingest copied text into MemOS.

    Runs in a daemon thread that polls the system clipboard at a configurable
    interval. When new text is detected (that is longer than MIN_CLIPBOARD_LENGTH),
    it is stored as a memory with source="clipboard".

    Example:
        watcher = ClipboardWatcher(engine)
        watcher.start()
        # ... user copies text, it gets auto-ingested ...
        watcher.stop()
    """

    def __init__(
        self,
        engine,
        poll_interval: float = POLL_INTERVAL,
        min_length: int = MIN_CLIPBOARD_LENGTH,
        max_length: int = MAX_CLIPBOARD_LENGTH,
    ) -> None:
        """Initialize the clipboard watcher.

        Args:
            engine:        The MemoryEngine to ingest into.
            poll_interval: Seconds between clipboard checks.
            min_length:    Minimum text length to trigger ingestion.
            max_length:    Maximum text length to ingest.
        """
        self._engine = engine
        self._poll_interval = poll_interval
        self._min_length = min_length
        self._max_length = max_length
        self._last_content: str = ""
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the clipboard watcher in a background daemon thread."""
        if self._running:
            return

        # Capture current clipboard content so we don't ingest
        # whatever was already there before starting
        try:
            self._last_content = pyperclip.paste() or ""
        except Exception:
            self._last_content = ""

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="memos-clipboard-watcher",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info(
            "ClipboardWatcher started (poll=%.1fs, min_len=%d)",
            self._poll_interval,
            self._min_length,
        )

    def stop(self) -> None:
        """Stop the clipboard watcher."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 1)
            self._thread = None
        self._running = False
        logger.info("ClipboardWatcher stopped.")

    def _watch_loop(self) -> None:
        """Main polling loop — runs in a daemon thread."""
        while not self._stop_event.is_set():
            try:
                current = pyperclip.paste() or ""

                # Check if clipboard content has changed
                if current != self._last_content:
                    self._last_content = current
                    stripped = current.strip()

                    # Only ingest if it meets our criteria
                    if (
                        len(stripped) >= self._min_length
                        and len(stripped) <= self._max_length
                    ):
                        self._ingest(stripped)

            except Exception as e:
                # pyperclip can throw on some platforms if clipboard is locked
                logger.debug("Clipboard read error (harmless): %s", e)

            # Wait for the next poll
            self._stop_event.wait(timeout=self._poll_interval)

    def _ingest(self, text: str) -> None:
        """Ingest clipboard text as a memory."""
        try:
            # Determine memory type based on content heuristics
            memory_type = self._detect_type(text)

            self._engine.store(
                content=text,
                source="clipboard",
                memory_type=memory_type,
                tags=["clipboard", "auto-captured"],
                metadata={"char_count": len(text)},
            )
            logger.info(
                "Ingested clipboard content (%d chars, type=%s)",
                len(text),
                memory_type,
            )
        except Exception as e:
            logger.warning("Failed to ingest clipboard content: %s", e)

    @staticmethod
    def _detect_type(text: str) -> str:
        """Heuristic detection of content type from clipboard text.

        Returns:
            A memory_type string like "code", "url", "note".
        """
        stripped = text.strip()

        # Check for code patterns
        code_indicators = [
            "def ", "class ", "import ", "from ", "function ", "const ",
            "let ", "var ", "return ", "if (", "for (", "while (",
            "public ", "private ", "static ", "void ", "int ",
            "=>", "->", "::", "#!/",
        ]
        if any(indicator in stripped for indicator in code_indicators):
            return "code"

        # Check for URLs
        if stripped.startswith(("http://", "https://", "ftp://")):
            return "url"

        # Check for file paths
        if stripped.startswith(("/", "C:\\", "~/")):
            return "file_path"

        # Default
        return "note"

    @property
    def is_running(self) -> bool:
        """Whether the watcher is currently running."""
        return self._running
