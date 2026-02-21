"""Embedding engine for MemOS.

Wraps Zvec's built-in DefaultLocalDenseEmbedding (sentence-transformers
all-MiniLM-L6-v2, 384 dimensions). The model is lazily loaded on first use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("memos.core.embeddings")


class EmbeddingEngine:
    """Generate vector embeddings from text using a local model.

    Uses Zvec's built-in DefaultLocalDenseEmbedding which wraps
    sentence-transformers (all-MiniLM-L6-v2, 384-dim, ~80 MB).

    The model is downloaded and loaded lazily on the first call to
    `embed_text()` — subsequent calls are instant.

    Example:
        >>> engine = EmbeddingEngine()
        >>> vec = engine.embed_text("Hello, world!")
        >>> len(vec)
        384
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._embedder = None  # Lazy-loaded

    def _ensure_loaded(self) -> None:
        """Lazily initialize the embedding model on first use."""
        if self._embedder is not None:
            return

        logger.info("Loading embedding model '%s' (first run may download ~80 MB)...", self._model_name)
        try:
            from zvec.extension import DefaultLocalDenseEmbedding

            self._embedder = DefaultLocalDenseEmbedding()
            logger.info("Embedding model loaded successfully (384 dimensions).")
        except ImportError:
            # Fallback: use sentence-transformers directly if zvec.extension
            # is not available (e.g., older zvec version)
            logger.warning("zvec.extension not available, falling back to sentence-transformers.")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._embedder = self._model
            logger.info("Fallback embedding model loaded.")

    def embed_text(self, text: str) -> list[float]:
        """Generate a vector embedding for a single text string.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the 384-dimensional embedding.
        """
        self._ensure_loaded()

        # Zvec's DefaultLocalDenseEmbedding has .embed()
        if hasattr(self._embedder, "embed"):
            vec = self._embedder.embed(text)
            return list(vec) if not isinstance(vec, list) else vec

        # Fallback: sentence-transformers .encode()
        vec = self._embedder.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.
        """
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors (384 for MiniLM)."""
        return 384
