import hashlib
import logging
import re
import threading
import time

import pyperclip

logger = logging.getLogger("memos.connectors.clipboard_watcher")

# Minimum length of clipboard text to trigger ingestion
MIN_CLIPBOARD_LENGTH = 30

# How often to poll the clipboard (seconds)
POLL_INTERVAL = 3.0

# Maximum clipboard text to ingest (100 KB)
MAX_CLIPBOARD_LENGTH = 102_400

# Patterns that look like sensitive data (passwords, keys, UUIDs)
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)api[-_]?key['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}['\"]?"), # API Keys
    re.compile(r"(?i)password['\"]?\s*[:=]\s*['\"]?.{4,}['\"]?"),                # Passwords
    re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"), # UUIDs
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),                                          # OpenAI-like keys
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),                         # Private keys
]


class ClipboardWatcher:
    """Passively watch the clipboard and auto-ingest copied text into MemOS.

    Includes strict guardrails:
    1. Minimum length (30+ characters).
    2. Sensitive data filtering (regex for API keys, passwords, UUIDs).
    3. Deduplication (SHA-256 hashing).
    """

    def __init__(
        self,
        engine,
        poll_interval: float = POLL_INTERVAL,
        min_length: int = MIN_CLIPBOARD_LENGTH,
        max_length: int = MAX_CLIPBOARD_LENGTH,
    ) -> None:
        self._engine = engine
        self._poll_interval = poll_interval
        self._min_length = min_length
        self._max_length = max_length
        self._last_hash: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the clipboard watcher in a background daemon thread."""
        if self._running:
            return

        # Fingerprint current clipboard so we don't ingest initial state
        try:
            initial = pyperclip.paste() or ""
            self._last_hash = self._fingerprint(initial)
        except Exception:
            self._last_hash = None

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

    def _fingerprint(self, text: str) -> str:
        """Generate a SHA-256 fingerprint of the text."""
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def _is_sensitive(self, text: str) -> bool:
        """Check if the text matches any sensitive data patterns."""
        return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)

    def _watch_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            try:
                current = pyperclip.paste() or ""
                stripped = current.strip()

                if not stripped:
                    self._stop_event.wait(timeout=self._poll_interval)
                    continue

                current_hash = self._fingerprint(stripped)

                # Check for changes via hash
                if current_hash != self._last_hash:
                    self._last_hash = current_hash

                    # Apply guardrails
                    if len(stripped) < self._min_length:
                        logger.debug("Clipboard too short (%d < %d)", len(stripped), self._min_length)
                    elif len(stripped) > self._max_length:
                        logger.debug("Clipboard too large (%d > %d)", len(stripped), self._max_length)
                    elif self._is_sensitive(stripped):
                        logger.warning("Clipboard content matched sensitive pattern, skipping.")
                    else:
                        self._ingest(stripped)

            except Exception as e:
                logger.debug("Clipboard read error: %s", e)

            self._stop_event.wait(timeout=self._poll_interval)

    def _ingest(self, text: str) -> None:
        """Ingest clipboard text as a memory."""
        try:
            memory_type = self._detect_type(text)
            self._engine.store(
                content=text,
                source="clipboard",
                memory_type=memory_type,
                tags=["clipboard", "auto-captured"],
                metadata={"char_count": len(text)},
            )
            logger.info("Ingested clipboard (type=%s, chars=%d)", memory_type, len(text))
        except Exception as e:
            logger.warning("Failed to ingest clipboard content: %s", e)

    @staticmethod
    def _detect_type(text: str) -> str:
        """Heuristic detection of content type."""
        stripped = text.strip()
        code_indicators = [
            "def ", "class ", "import ", "from ", "function ", "const ",
            "let ", "var ", "return ", "if (", "for (", "while (",
            "public ", "private ", "static ", "void ", "int ",
            "=>", "->", "::", "#!/",
        ]
        if any(indicator in stripped for indicator in code_indicators):
            return "code"
        if stripped.startswith(("http://", "https://", "ftp://")):
            return "url"
        return "note"

    @property
    def is_running(self) -> bool:
        return self._running
