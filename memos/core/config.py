"""Configuration for MemOS daemon and services."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    """Return the default MemOS data directory (~/.memos/)."""
    return Path(os.environ.get("MEMOS_DATA_DIR", Path.home() / ".memos"))


@dataclass
class MemOSConfig:
    """Central configuration for the MemOS daemon.

    Attributes:
        data_dir:       Root directory for all MemOS data.
        api_port:       Port for the REST API server.
        api_host:       Host to bind the REST API to.
        backend:        Which vector store backend to use ("zvec" | "lancedb").
        embedding_model: Name of the sentence-transformers model for embeddings.
        embedding_dim:  Dimensionality of the embedding vectors.
        log_level:      Logging level (DEBUG, INFO, WARNING, ERROR).
    """

    data_dir: Path = field(default_factory=_default_data_dir)
    api_port: int = 11437
    api_host: str = "127.0.0.1"
    backend: str = "lancedb" if os.name == "nt" else "zvec"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    log_level: str = "INFO"

    # --- Derived paths ---

    @property
    def db_path(self) -> Path:
        """Path to the vector database storage."""
        return self.data_dir / "db"

    @property
    def memories_path(self) -> Path:
        """Path to the memories collection."""
        return self.db_path / "memories"

    @property
    def entities_path(self) -> Path:
        """Path to the knowledge graph entities collection."""
        return self.db_path / "entities"

    @property
    def relationships_path(self) -> Path:
        """Path to the knowledge graph relationships collection."""
        return self.db_path / "relationships"

    @property
    def pid_file(self) -> Path:
        """Path to the daemon PID file."""
        return self.data_dir / "memos.pid"

    @property
    def log_file(self) -> Path:
        """Path to the daemon log file."""
        return self.data_dir / "memos.log"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.mkdir(parents=True, exist_ok=True)
